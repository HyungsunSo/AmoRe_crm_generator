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


STYLE_TYPES = [
    'Time_Urgency_Style',
    'Information_Universal_Style',
    'FOMO_Psychology_Style',
    'Emotional_Style',
    'Seasonal_Style',
    'Mixed_Strategies'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona', required=True, help='페르소나 인덱스(0~) 또는 이름')
    parser.add_argument('--brand', required=True, help='브랜드명 (brand_stories.json 키)')
    parser.add_argument('--product', required=True, help='제품명(부분 일치 가능)')
    parser.add_argument('--stage_index', type=int, required=True, help='CRM 목적 인덱스 (0~4)')
    parser.add_argument('--top_k', type=int, default=5, help='CRM RAG Top-K 및 리뷰 Top-K')
    parser.add_argument('--qwen_model', default='Qwen/Qwen2.5-1.5B-Instruct', help='Qwen 로컬 모델 이름')
    parser.add_argument('--exa_model', default='LGAI-EXAONE/EXAONE-4.0-1.2B', help='Exaone 로컬 모델 이름')
    parser.add_argument('--is_event', type=int, default=0, help='캠페인 이벤트 포함 여부 (0 or 1)')
    parser.add_argument('--style_index', type=int, default=0, help='CRM 템플릿 스타일 인덱스 (0:시간긴박, 1:정보전달, 2:FOMO, 3:감성, 4:시즌, 5:믹스)')
    parser.add_argument('--out_path', default=None, help='결과 저장 경로')
    args = parser.parse_args()

    # 인덱스를 기반으로 AARRR 스테이지 문자열 설정
    if 0 <= args.stage_index < len(STAGE_ORDER):
        aarrr_stage = STAGE_ORDER[args.stage_index]
    else:
        aarrr_stage = STAGE_ORDER[0]
        print(f"[경고] 잘못된 stage_index. 기본값({aarrr_stage})으로 진행합니다.")

    # 인덱스를 스타일 키로 변환
    if 0 <= args.style_index < len(STYLE_TYPES):
        style_type = STYLE_TYPES[args.style_index]
    else:
        style_type = STYLE_TYPES[0]
        print(f"[경고] 잘못된 style_index. 기본값({style_type})으로 진행합니다.")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')

    # 데이터 로드
    personas = load_json(os.path.join(data_dir, 'personas.json'))
    products = load_json(os.path.join(data_dir, 'products.json'))
    brand_stories = load_json(os.path.join(data_dir, 'brand_stories.json'))
    crm_goals = load_json(os.path.join(data_dir, 'crm_goals.json'))
    crm_categorized = load_json(os.path.join(data_dir, 'crm_analysis_results_categorized.json'))
    campaign_events = load_json(os.path.join(data_dir, 'campaign_events.json'))
    integrated_templates = load_json(os.path.join(data_dir, 'integrated_crm_templates.json'))
    # FOMO samples will be handled by style_index if selected
    # No more global fomo_data loading here

    persona = find_persona(personas, args.persona)
    product = find_product(products, args.brand, args.product)

    # Qwen 하이라이트 준비
    highlights = top_highlights_for_product(persona, product, top_k=args.top_k)
    highlight_texts = [h['snippet'] for h in highlights]

    timeline = []

    # 캠페인 이벤트 선택 로직
    selected_event = None
    if args.is_event == 1:
        stage_events = campaign_events.get(aarrr_stage, {})
        promo_y_list = stage_events.get("promotion_y", [])
        if promo_y_list:
            selected_event = random.choice(promo_y_list)

    # Qwen 생성
    qwen_start = time.time()
    q_generator = LocalQwenGenerator(model_name=args.qwen_model)
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

    # Exaone 톤 보정 준비 (RAG 컨텍스트)
    brand_story = pick_brand_story(brand_stories, args.brand)
    crm_goal = load_crm_goal_meta(crm_goals, args.stage_index)
    bucket = select_stage_bucket(crm_categorized, args.stage_index)
    crm_snippets = rag_crm_snippets(bucket, q_draft[:500], top_k=args.top_k)

    # CRM 템플릿 검색 로직 (Exaone 컨텍스트)
    selected_templates = []
    style_data = integrated_templates.get(style_type, {}).get("content", {})
    
    # 템플릿 구조 처리 (List vs Dict)
    start_time = time.time()
    
    # 딕셔너리 내부 키/리스트 탐색
    # 구조가 복잡하므로 재귀보단 단순 분기 처리
    candidates_pool = []
    
    # CASE A: Dict-based (e.g. Information_Universal_Style -> CRM_Universal_Templates_50 -> "1_Acquisition...")
    # 템플릿 JSON의 "content" 바로 아래에 카테고리(예: CRM_Universal_Templates_50)가 있고 그 안에 stage key가 있는 구조라고 가정
    # 또는 바로 stage list가 있을 수도 있음.
    
    # 일관성을 위해 content의 values를 순회
    for group_key, group_val in style_data.items():
        if isinstance(group_val, dict):
             # Dict-based stages (e.g. "1_Acquisition_획득": [...])
             for st_key, st_list in group_val.items():
                 if aarrr_stage in st_key:
                     if isinstance(st_list, list):
                         candidates_pool.extend(st_list)
        elif isinstance(group_val, list):
            # List-based stages (e.g. CRM_Urgency_TimeLimit_Templates -> [ {stage: "...", data: [...]}, ...])
            for item in group_val:
                if isinstance(item, dict) and 'stage' in item and 'data' in item:
                    if aarrr_stage in item['stage']:
                         candidates_pool.extend(item['data'])

    if candidates_pool:
        # 랜덤 2~3개 선택
        k = min(len(candidates_pool), random.randint(2, 3))
        selected_templates = random.sample(candidates_pool, k)

    # 템플릿을 문자열로 변환하여 프롬프트에 추가
    style_ref_templates = []
    for t in selected_templates:
        # style, title, content, cta 필드 조합
        t_str = f"스타일: {t.get('style','')}\n제목: {t.get('title','')}\n본문: {t.get('content','')}\nCTA: {t.get('cta','')}"
        style_ref_templates.append(t_str)

    exa_messages = build_exaone_prompt(
        qwen_draft=q_draft,
        persona=persona,
        brand_story=brand_story,
        crm_goal=crm_goal,
        stage_index=args.stage_index,
        crm_snippets=crm_snippets,
        style_examples=style_ref_templates  # fomo_examples 대신 style_examples로 전달
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

    # 결과 저장 (로그용 전체 데이터)
    log_dir = os.path.join(base, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_filename = f"pipeline_{args.brand}_{STAGE_ORDER[args.stage_index]}_{timestamp_str}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 사용자 요청 결과물 저장 (심플 데이터)
    final_output_dir = os.path.join(base, 'outputs')
    os.makedirs(final_output_dir, exist_ok=True)
    
    # 요청된 필드 매핑
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

    print("✓ 로그 저장 완료:", log_path)
    print("✓ 결과 저장 완료:", output_path)
    print("--- Qwen 초안 ---")
    print(q_draft[:400])
    print("\n--- Exaone 보정 ---")
    print(exa_output[:400])
    print("\n--- Exaone 입력 프롬프트 ---")
    print(exa_prompt_text[:1200])


if __name__ == '__main__':
    main()
