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
    return highlights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona', required=True, help='페르소나 인덱스(0~) 또는 이름')
    parser.add_argument('--brand', required=True, help='브랜드명 (brand_stories.json 키)')
    parser.add_argument('--product', required=True, help='제품명(부분 일치 가능)')
    parser.add_argument('--stage_index', type=int, required=True, help='CRM 목적 인덱스 (0~4)')
    parser.add_argument('--top_k', type=int, default=5, help='CRM RAG Top-K 및 리뷰 Top-K')
    parser.add_argument('--qwen_model', default='Qwen/Qwen2.5-1.5B-Instruct', help='Qwen 로컬 모델 이름')
    parser.add_argument('--exa_model', default='LGAI-EXAONE/EXAONE-4.0-1.2B', help='Exaone 로컬 모델 이름')
    parser.add_argument('--out_path', default=None, help='결과 저장 경로')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')

    # 데이터 로드
    personas = load_json(os.path.join(data_dir, 'personas.json'))
    products = load_json(os.path.join(data_dir, 'products.json'))
    brand_stories = load_json(os.path.join(data_dir, 'brand_stories.json'))
    crm_goals = load_json(os.path.join(data_dir, 'crm_goals.json'))
    crm_categorized = load_json(os.path.join(data_dir, 'crm_analysis_results_categorized.json'))
    fomo_data = load_json(os.path.join(data_dir, 'FOMO.json'))

    persona = find_persona(personas, args.persona)
    product = find_product(products, args.brand, args.product)

    # Qwen 하이라이트 준비
    highlights = top_highlights_for_product(persona, product, top_k=args.top_k)
    highlight_texts = [h['snippet'] for h in highlights]

    timeline = []

    # Qwen 생성
    qwen_start = time.time()
    q_generator = LocalQwenGenerator(model_name=args.qwen_model)
    q_draft, q_dur = q_generator.generate_marketing_draft(
        product.get('brand_name', ''),
        product.get('name', ''),
        persona,
        product.get('reviews', []),
        highlight_texts
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

    # Exaone 톤 보정 준비 (RAG 컨텍스트)
    brand_story = pick_brand_story(brand_stories, args.brand)
    crm_goal = load_crm_goal_meta(crm_goals, args.stage_index)
    bucket = select_stage_bucket(crm_categorized, args.stage_index)
    crm_snippets = rag_crm_snippets(bucket, q_draft[:500], top_k=args.top_k)
    fomo_examples = format_fomo_examples(fomo_data, args.stage_index, limit=3)

    exa_messages = build_exaone_prompt(
        qwen_draft=q_draft,
        persona=persona,
        brand_story=brand_story,
        crm_goal=crm_goal,
        stage_index=args.stage_index,
        crm_snippets=crm_snippets,
        fomo_examples=fomo_examples,
    )
    # 로그용 전체 프롬프트 텍스트
    exa_prompt_text = "\n\n".join([f"[{m.get('role','')}] {m.get('content','')}" for m in exa_messages])

    # Exaone 실행
    exa_start = time.time()
    exa_generator = ExaoneToneCorrector(model_name=args.exa_model)
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

    # 결과 저장
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
            "fomo_examples": fomo_examples,
            "result_raw": exa_output
        },
        "timeline": timeline
    }

    out_dir = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out_path or os.path.join(
        out_dir,
        f"pipeline_{args.brand}_{STAGE_ORDER[args.stage_index]}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("✓ 파이프라인 완료:", out_path)
    print("--- Qwen 초안 ---")
    print(q_draft[:400])
    print("\n--- Exaone 보정 ---")
    print(exa_output[:400])
    print("\n--- Exaone 입력 프롬프트 ---")
    print(exa_prompt_text[:1200])


if __name__ == '__main__':
    main()
