#!/usr/bin/env python3
"""
Qwen → Exaone CRM 톤 보정 파이프라인 (원커맨드)

1) Qwen 로컬 모델로 마케팅 초안 생성
2) CRM RAG + FOMO + 브랜드 스토리 + CRM 목표 컨텍스트로 Exaone 톤 보정
3) 타임라인과 함께 JSON 저장

예시:
  python3 src/run_qwen_exaone_pipeline.py \\
    --persona 0 \\
    --brand 설화수 \\
    --product "자음생크림 리치 단품세트" \\
    --stage_index 1
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from rag_utils import build_persona_query, extract_candidate_texts, extract_highlight_snippet, vectorize_texts, cosine  # noqa: E402
from generate_marketing import LocalQwenGenerator, find_persona, find_product, load_json  # noqa: E402
from tone_correction import (  # noqa: E402
    build_exaone_prompt,
    ExaoneToneCorrector,
    pick_brand_story,
    load_crm_goal_meta,
    select_stage_bucket,
    rag_crm_snippets,
    format_fomo_examples,
    STAGE_ORDER,
)


def top_highlights_for_product(persona, product, top_k=3):
    cache_key = _highlight_cache_key(persona, product, top_k)
    cached = _HIGHLIGHT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    query = build_persona_query(persona)
    candidates = extract_candidate_texts(product)
    all_texts = [query] + candidates
    vectors = vectorize_texts(all_texts)
    if vectors is None or len(vectors) == 0:
        return []
    q_vec = vectors[0]
    cand_vecs = vectors[1:]

    scores = [(cosine(q_vec, v), i, candidates[i]) for i, v in enumerate(cand_vecs)]
    scores.sort(reverse=True, key=lambda x: x[0])
    top = scores[:top_k]

    highlights = []
    for score, idx, text in top:
        highlights.append(
            {
                "score": float(score),
                "index": idx,
                "text": text,
                "snippet": extract_highlight_snippet(text),
            }
        )
    _HIGHLIGHT_CACHE[cache_key] = highlights
    return highlights


STYLE_TYPES = [
    'Time_Urgency_Style',
    'Information_Universal_Style',
    'FOMO_Psychology_Style',
    'Emotional_Style',
    'Seasonal_Style',
    'Mixed_Strategies'
]

_QWEN_GENERATOR_CACHE = {}
_EXAONE_GENERATOR_CACHE = {}
_STYLE_POOL_CACHE = {}
_HIGHLIGHT_CACHE = {}


def _get_qwen_generator(model_name):
    cached = _QWEN_GENERATOR_CACHE.get(model_name)
    if cached:
        return cached
    generator = LocalQwenGenerator(model_name=model_name)
    _QWEN_GENERATOR_CACHE[model_name] = generator
    return generator


def _get_exaone_generator(model_name):
    cached = _EXAONE_GENERATOR_CACHE.get(model_name)
    if cached:
        return cached
    generator = ExaoneToneCorrector(model_name=model_name)
    _EXAONE_GENERATOR_CACHE[model_name] = generator
    return generator


def _highlight_cache_key(persona, product, top_k):
    persona_key = persona.get("name") if isinstance(persona, dict) else str(persona)
    product_key = None
    if isinstance(product, dict):
        product_key = product.get("product_id") or product.get("name")
    if not product_key:
        product_key = str(product)
    return persona_key, product_key, top_k


def _get_style_candidates(style_data, aarrr_stage):
    key = (id(style_data), aarrr_stage)
    cached = _STYLE_POOL_CACHE.get(key)
    if cached is not None:
        return cached

    candidates_pool = []
    for group_val in style_data.values():
        if isinstance(group_val, dict):
            for st_key, st_list in group_val.items():
                if aarrr_stage in st_key and isinstance(st_list, list):
                    candidates_pool.extend(st_list)
        elif isinstance(group_val, list):
            for item in group_val:
                if isinstance(item, dict) and 'stage' in item and 'data' in item:
                    if aarrr_stage in item['stage']:
                        candidates_pool.extend(item['data'])

    _STYLE_POOL_CACHE[key] = candidates_pool
    return candidates_pool

def _load_data(base):
    data_dir = os.path.join(base, 'data')
    return {
        "personas": load_json(os.path.join(data_dir, 'personas.json')),
        "products": load_json(os.path.join(data_dir, 'products.json')),
        "brand_stories": load_json(os.path.join(data_dir, 'brand_stories.json')),
        "crm_goals": load_json(os.path.join(data_dir, 'crm_goals.json')),
        "crm_categorized": load_json(os.path.join(data_dir, 'crm_analysis_results_categorized.json')),
        "campaign_events": load_json(os.path.join(data_dir, 'campaign_events.json')),
        "integrated_templates": load_json(os.path.join(data_dir, 'integrated_crm_templates.json')),
    }


def _normalize_row(row):
    if not isinstance(row, dict):
        raise ValueError('Batch row must be a dict')

    def _to_int(value, default=None):
        if value is None:
            return default
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if text == '':
            return default
        return int(text)

    def _to_bool_int(value, default=0):
        if isinstance(value, bool):
            return 1 if value else 0
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return 1 if int(value) != 0 else 0
        text = str(value).strip().lower()
        if text in {'1', 'true', 'yes', 'y', 't'}:
            return 1
        if text in {'0', 'false', 'no', 'n', 'f'}:
            return 0
        return default

    normalized = dict(row)
    if 'stage_index' in normalized:
        normalized['stage_index'] = _to_int(normalized.get('stage_index'))
    if 'style_index' in normalized:
        normalized['style_index'] = _to_int(normalized.get('style_index'), default=0)
    if 'is_event' in normalized:
        normalized['is_event'] = _to_bool_int(normalized.get('is_event'), default=0)
    return normalized


def _run_pipeline(args, data=None, q_generator=None, exa_generator=None):
    # Resolve stage and style indices with fallbacks
    if 0 <= args.stage_index < len(STAGE_ORDER):
        aarrr_stage = STAGE_ORDER[args.stage_index]
    else:
        aarrr_stage = STAGE_ORDER[0]
        print(f"[WARN] Invalid stage_index. Using default ({aarrr_stage}).")

    if 0 <= args.style_index < len(STYLE_TYPES):
        style_type = STYLE_TYPES[args.style_index]
    else:
        style_type = STYLE_TYPES[0]
        print(f"[WARN] Invalid style_index. Using default ({style_type}).")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data is None:
        data = _load_data(base)

    personas = data['personas']
    products = data['products']
    brand_stories = data['brand_stories']
    crm_goals = data['crm_goals']
    crm_categorized = data['crm_categorized']
    campaign_events = data['campaign_events']
    integrated_templates = data['integrated_templates']

    persona = find_persona(personas, args.persona)
    product = find_product(products, args.brand, args.product)

    # Qwen highlights
    highlights = top_highlights_for_product(persona, product, top_k=args.top_k)
    highlight_texts = [h['snippet'] for h in highlights]

    timeline = []

    # Optionally select a campaign event
    selected_event = None
    if args.is_event == 1:
        stage_events = campaign_events.get(aarrr_stage, {})
        promo_y_list = stage_events.get("promotion_y", [])
        if promo_y_list:
            selected_event = random.choice(promo_y_list)

    # Qwen draft
    qwen_start = time.time()
    if q_generator is None:
        q_generator = _get_qwen_generator(args.qwen_model)
    q_draft, q_dur = q_generator.generate_marketing_draft(
        product.get('brand_name', ''),
        product.get('name', ''),
        persona,
        product.get('reviews', []),
        highlight_texts,
        campaign_event_info=selected_event
    )
    qwen_end = time.time()
    timeline.append({
        "step": "qwen_generation",
        "model": args.qwen_model,
        "started_at": datetime.fromtimestamp(qwen_start, timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(qwen_end, timezone.utc).isoformat(),
        "duration_seconds": q_dur if q_dur is not None else (qwen_end - qwen_start),
        "output_raw": q_draft
    })

    # Exaone prompt inputs (with RAG snippets)
    brand_story = pick_brand_story(brand_stories, args.brand)
    crm_goal = load_crm_goal_meta(crm_goals, args.stage_index)
    bucket = select_stage_bucket(crm_categorized, args.stage_index)
    crm_snippets = rag_crm_snippets(bucket, q_draft[:500], top_k=args.top_k)

    # Pick CRM style templates for Exaone
    selected_templates = []
    style_data = integrated_templates.get(style_type, {}).get("content", {})
    candidates_pool = _get_style_candidates(style_data, aarrr_stage)

    if candidates_pool:
        # Sample 2-3 templates
        k = min(len(candidates_pool), random.randint(2, 3))
        selected_templates = random.sample(candidates_pool, k)

    # Build template reference strings
    style_ref_templates = []
    for t in selected_templates:
        t_str = (
            f"Style: {t.get('style','')}\n"
            f"Title: {t.get('title','')}\n"
            f"Body: {t.get('content','')}\n"
            f"CTA: {t.get('cta','')}"
        )
        style_ref_templates.append(t_str)

    exa_messages = build_exaone_prompt(
        qwen_draft=q_draft,
        persona=persona,
        brand_story=brand_story,
        crm_goal=crm_goal,
        stage_index=args.stage_index,
        crm_snippets=crm_snippets,
        style_examples=style_ref_templates
    )
    # Flatten prompt for logging
    exa_prompt_text = "\n\n".join(
        [f"[{m.get('role','')}] {m.get('content','')}" for m in exa_messages]
    )

    # Exaone generation
    exa_start = time.time()
    if exa_generator is None:
        exa_generator = _get_exaone_generator(args.exa_model)
    exa_output = exa_generator.generate(exa_messages)
    exa_end = time.time()
    timeline.append({
        "step": "exaone_prompt",
        "model": args.exa_model,
        "prompt_preview": exa_prompt_text[:800]
    })
    timeline.append({
        "step": "exaone_tone_correction",
        "model": args.exa_model,
        "started_at": datetime.fromtimestamp(exa_start, timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(exa_end, timezone.utc).isoformat(),
        "duration_seconds": exa_end - exa_start,
        "output_raw": exa_output
    })

    # Build output
    out = {
        "persona_input": args.persona,
        "persona_profile": persona,
        "brand": args.brand,
        "product_query": args.product,
        "product_basic": {
            "product_id": product.get('product_id'),
            "name": product.get('name'),
            "brand_name": product.get('brand_name'),
            "price": product.get('price'),
            "url": product.get('url')
        },
        "stage_index": args.stage_index,
        "stage_name": STAGE_ORDER[args.stage_index],
        "stage_kr": crm_goal.get('stage_kr', ''),
        "objective": crm_goal.get('objective', ''),
        "target_state": crm_goal.get('target_state', ''),
        "style_index": args.style_index,
        "style_type": style_type,
        "style_templates": style_ref_templates,
        "is_event": True if args.is_event == 1 else False,
        "selected_event": selected_event,
        "qwen": {
            "model": args.qwen_model,
            "draft": q_draft,
            "highlights": highlights
        },
        "exaone": {
            "model": args.exa_model,
            "prompt_messages": exa_messages,
            "prompt_text": exa_prompt_text,
            "rag_crm_snippets": crm_snippets,
            "selected_style_templates": style_ref_templates,
            "result_raw": exa_output
        },
        "timeline": timeline
    }

    # Write log output
    log_dir = os.path.join(base, 'log')
    os.makedirs(log_dir, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_filename = f"pipeline_{args.brand}_{STAGE_ORDER[args.stage_index]}_{timestamp_str}.json"
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # Write lightweight output
    final_output_dir = os.path.join(base, 'outputs')
    os.makedirs(final_output_dir, exist_ok=True)

    # Minimal result for user
    user_output = {
        "persona_id": args.persona,
        "product_id": product.get('product_id'),
        "send_purpose": STAGE_ORDER[args.stage_index],
        "customer_segment": persona.get('name'),
        "has_event": True if args.is_event == 1 else False,
        "marketing_draft": q_draft,
        "crm_message": exa_output
    }
    if selected_event:
        user_output["event_info"] = selected_event

    output_filename = f"result_{args.brand}_{timestamp_str}.json"
    output_path = args.out_path or os.path.join(final_output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(user_output, f, ensure_ascii=False, indent=2)

    print("Log saved:", log_path)
    print("Result saved:", output_path)
    print("--- Qwen draft ---")
    print(q_draft[:400])
    print("\n--- Exaone refined ---")
    print(exa_output[:400])
    print("\n--- Exaone prompt ---")
    print(exa_prompt_text[:1200])

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona', required=False, help='Persona index (0~) or name')
    parser.add_argument('--brand', required=False, help='Brand name (brand_stories.json key)')
    parser.add_argument('--product', required=False, help='Product name (partial match allowed)')
    parser.add_argument('--stage_index', type=int, required=False, help='CRM stage index (0~4)')
    parser.add_argument('--top_k', type=int, default=5, help='RAG Top-K and review Top-K')
    parser.add_argument('--qwen_model', default='Qwen/Qwen2.5-1.5B-Instruct', help='Qwen local model name')
    parser.add_argument('--exa_model', default='LGAI-EXAONE/EXAONE-4.0-1.2B', help='Exaone local model name')
    parser.add_argument('--is_event', type=int, default=0, help='Include campaign event (0 or 1)')
    parser.add_argument('--style_index', type=int, default=0, help='CRM template style index (0~5)')
    parser.add_argument('--out_path', default=None, help='Output path')
    parser.add_argument('--batch_json', default=None, help='Batch input JSON path (list of rows)')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.batch_json:
        with open(args.batch_json, 'r', encoding='utf-8') as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError('batch_json must be a list of row dicts')

        data = _load_data(base)
        q_generator = _get_qwen_generator(args.qwen_model)
        exa_generator = _get_exaone_generator(args.exa_model)
        outputs = []

        for idx, row in enumerate(rows, start=1):
            normalized = _normalize_row(row)
            row_args = argparse.Namespace(**vars(args))
            for key, value in normalized.items():
                if hasattr(row_args, key):
                    setattr(row_args, key, value)
            if row_args.persona is None or row_args.brand is None or row_args.product is None or row_args.stage_index is None:
                raise ValueError(f"Missing required fields in batch row {idx}")
            outputs.append(_run_pipeline(row_args, data=data, q_generator=q_generator, exa_generator=exa_generator))

        if args.out_path:
            with open(args.out_path, 'w', encoding='utf-8') as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
        return outputs

    # Single run
    required = ['persona', 'brand', 'product', 'stage_index']
    for key in required:
        if getattr(args, key) is None:
            parser.error(f"--{key} is required unless --batch_json is provided")

    data = _load_data(base)
    return _run_pipeline(args, data=data)
if __name__ == '__main__':
    main()
