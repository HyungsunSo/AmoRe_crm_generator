#!/usr/bin/env python3
"""
SLM-Optimized CRM Pipeline Runner v2

steps_v2.py의 6단계 파이프라인을 실행합니다.
- 로컬 GGUF 모델 또는 Ollama API와 함께 작동
- 스타일 요소 주입 기반 브랜드 톤 적용
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from sequential_pipeline.steps_v2 import (
    ReviewSummarizer,
    BriefGenerator,
    PersonaWriter,
    GoalSetter,
    BrandStyler,
    FinalPolisher,
)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


from rag_utils import build_persona_query, extract_candidate_texts, extract_highlight_snippet, vectorize_texts, cosine
from generate_marketing import find_persona, find_product, load_json


STAGE_ORDER = ["Acquisition", "Activation", "Retention", "Revenue", "Referral"]


def top_highlights_for_product(persona, product, top_k=3):
    """페르소나와 제품 정보 기반 RAG 하이라이트 추출"""
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
        highlights.append({
            "score": float(score),
            "index": idx,
            "text": text,
            "snippet": extract_highlight_snippet(text),
        })
    return highlights


def main():
    parser = argparse.ArgumentParser(description='SLM-Optimized 6-Stage CRM Pipeline')
    parser.add_argument('--persona', required=True, help='페르소나 인덱스(0~) 또는 이름')
    parser.add_argument('--brand', required=True, help='브랜드명 (에뛰드, 설화수, 이니스프리, 라네즈, 헤라, 에스트라)')
    parser.add_argument('--product', required=True, help='제품명 (부분 일치 가능)')
    parser.add_argument('--stage_index', type=int, required=True, help='CRM 목적 인덱스 (0=Acquisition, 1=Activation, 2=Retention, 3=Revenue, 4=Referral)')
    parser.add_argument('--top_k', type=int, default=3, help='리뷰 Top-K')
    parser.add_argument('--out_dir', default=None, help='출력 디렉토리')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, 'data')
    
    # 인덱스를 기반으로 AARRR 스테이지 문자열 설정
    if 0 <= args.stage_index < len(STAGE_ORDER):
        aarrr_stage = STAGE_ORDER[args.stage_index]
    else:
        aarrr_stage = STAGE_ORDER[0]
        print(f"[경고] 잘못된 stage_index. 기본값({aarrr_stage})으로 진행합니다.")

    # 데이터 로드
    personas = load_json(os.path.join(data_dir, 'personas.json'))
    products = load_json(os.path.join(data_dir, 'products.json'))
    
    persona = find_persona(personas, args.persona)
    product = find_product(products, args.brand, args.product)
    persona_name = persona.get('name', 'default')
    product_name = product.get('name', args.product)
    
    # 출력 디렉토리 설정
    out_dir = args.out_dir or os.path.join(base, 'outputs', 'slm_v2_logs')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[SLM v2 Pipeline] {args.brand} - {product_name}")
    print(f"[Persona] {persona_name} | [Stage] {aarrr_stage}")
    print(f"{'='*60}\n")

    timeline = []
    total_start = time.time()

    # === Step 0: 하이라이트 추출 (RAG) ===
    print("[Step 0/7] 하이라이트 추출 중...")
    highlights_data = top_highlights_for_product(persona, product, top_k=args.top_k)
    highlight_texts = [h['snippet'] for h in highlights_data]
    highlights_str = ', '.join(highlight_texts) if highlight_texts else f"{product_name} 관련 핵심 특징"
    print(f"  ✓ 추출된 하이라이트: {highlights_str[:100]}...\n")

    # === Step 1: ReviewSummarizer (긍정 키워드 추출) ===
    print("[Step 1/7] 리뷰에서 긍정 키워드 추출 중...")
    positive_kw, dur0 = ReviewSummarizer().run(highlights_str)
    timeline.append({"step": "ReviewSummarizer", "duration": dur0})
    print(f"  ✓ 완료 ({dur0:.1f}s)")
    print(f"  ✨ 긍정 키워드: {positive_kw[:80]}...\n")

    # === Step 2: BriefGenerator (키워드 정리) ===
    print("[Step 2/7] 키워드 정리 중...")
    keywords1, dur1 = BriefGenerator().run(product_name, positive_kw)
    timeline.append({"step": "BriefGenerator", "duration": dur1})
    print(f"  ✓ 완료 ({dur1:.1f}s)")
    print(f"  📝 Keywords: {keywords1[:100]}...\n")

    # === Step 3: PersonaWriter (감정 키워드 추가) ===
    print("[Step 3/7] 감정 키워드 추가 중...")
    keywords2, dur2 = PersonaWriter().run(keywords1, persona_name)
    timeline.append({"step": "PersonaWriter", "duration": dur2})
    print(f"  ✓ 완료 ({dur2:.1f}s)")
    print(f"  👤 Keywords: {keywords2[:100]}...\n")

    # === Step 4: GoalSetter (CTA 키워드 추가) ===
    print("[Step 4/7] CTA 키워드 추가 중...")
    keywords3, dur3 = GoalSetter().run(keywords2, aarrr_stage)
    timeline.append({"step": "GoalSetter", "duration": dur3})
    print(f"  ✓ 완료 ({dur3:.1f}s)")
    print(f"  🎯 Keywords: {keywords3[:100]}...\n")

    # === Step 5: BrandStyler (문장 조합) ===
    print("[Step 5/7] 브랜드 스타일 문장 생성 중...")
    styled, dur4 = BrandStyler().run(keywords3, args.brand)
    timeline.append({"step": "BrandStyler", "duration": dur4})
    print(f"  ✓ 완료 ({dur4:.1f}s)")
    print(f"  🎨 Styled: {styled}\n")

    # === Step 6: FinalPolisher (본문 윤문 - 클로바) ===
    print("[Step 6/6] 본문 윤문 중 (클로바)...")
    body, dur5 = FinalPolisher().run(styled)
    timeline.append({"step": "FinalPolisher", "duration": dur5})
    print(f"  ✓ 완료 ({dur5:.1f}s)")
    print(f"  ✨ Body: {body}\n")

    total_duration = time.time() - total_start

    # === 결과 저장 ===
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = {
        "brand": args.brand,
        "product": product_name,
        "persona": persona_name,
        "stage": aarrr_stage,
        "pipeline": "SLM-Optimized v2 (Keyword-Based, 6-Stage)",
        "steps": {
            "positive_keywords": positive_kw,
            "keywords_1_brief": keywords1,
            "keywords_2_persona": keywords2,
            "keywords_3_cta": keywords3,
            "styled_message": styled,
            "body": body,
        },
        "final_output": body,
        "timeline": timeline,
        "total_duration_seconds": total_duration,
        "timestamp": timestamp,
    }

    out_path = os.path.join(out_dir, f"slm_v2_{args.brand}_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"{'='*60}")
    print(f"✓ 완료! 총 소요시간: {total_duration:.1f}초")
    print(f"✓ 결과 저장: {out_path}")
    print(f"{'='*60}")
    print(f"\n🎉 최종 CRM 메시지: {body}")


if __name__ == '__main__':
    main()
