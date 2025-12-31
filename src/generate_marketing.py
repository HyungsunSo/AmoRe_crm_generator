#!/usr/bin/env python3
"""
Persona-Product RAG 및 로컬 Qwen 모델 기반 마케팅 초안 생성기

사용법:
  python3 src/generate_marketing.py --persona 0 --brand 설화수 --product "자음생크림 리치 단품세트"
  python3 src/generate_marketing.py --persona 0 --brand 설화수 --product "자음생크림 리치 단품세트" --use_local_model
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from rag_utils import vectorize_texts, cosine, extract_candidate_texts, extract_highlight_snippet, build_persona_query


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_persona(personas, persona_input):
    # persona_input can be index or name
    try:
        idx = int(persona_input)
        return personas[idx]
    except Exception:
        # lookup by name
        for p in personas:
            if p.get('name', '').lower() == persona_input.lower():
                return p
    raise ValueError('페르소나를 찾을 수 없습니다: %s' % persona_input)


def find_product(products, brand, product_name):
    brand_l = (brand or '').strip().lower()
    name_l = (product_name or '').strip().lower()
    # prefer exact brand + name match, otherwise partial
    for p in products:
        if p.get('brand_name','').strip().lower() == brand_l and name_l in p.get('name','').strip().lower():
            return p
    for p in products:
        if brand_l in p.get('brand_name','').strip().lower() or name_l in p.get('name','').strip().lower():
            return p
    raise ValueError('제품을 찾을 수 없습니다: %s / %s' % (brand, product_name))


# Ensure src directory is importable when running script directly
sys.path.insert(0, os.path.dirname(__file__))
from rag_utils import vectorize_texts, cosine, is_positive_review, extract_candidate_texts, extract_highlight_snippet, build_persona_query


def get_device():
    """Get appropriate device for model inference."""
    # MPS는 생성 작업에서 문제가 있을 수 있으므로 CUDA만 사용하고 나머지는 CPU 사용
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalQwenGenerator:
    """로컬 Qwen 모델을 사용하여 마케팅 초안을 생성합니다."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = get_device()
        self.model_name = model_name
        print(f"[로컬 Qwen] 디바이스: {self.device}")
        print(f"[로컬 Qwen] 모델 로딩 중: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        try:
            self.model.eval()
        except Exception:
            pass
        print("[로컬 Qwen] 모델 로딩 완료")
    
    def generate_text(self, messages, max_tokens=512, temperature=0.1):
        """Generate text using the local model."""
        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            input_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        t_start = time.time()
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except AttributeError:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        t_end = time.time()
        
        generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip(), t_end - t_start
    
    def generate_marketing_draft(self, brand_name, product_name, persona, reviews, highlights):
        """생성: 마케팅 초안 (One-Stage)."""
        persona_traits = ", ".join(persona.get("traits", []) if isinstance(persona.get("traits"), list) else [])
        review_text = "\n".join([f"- {r.get('text', '')[:150]}" for r in reviews[:3]])
        highlights_text = "\n".join([f"- {h}" for h in highlights[:3]])
        
        prompt = f"""당신은 마케팅 카피라이터입니다. 아래 정보를 바탕으로 마케팅 초안을 작성하세요.

[제품]
브랜드: {brand_name}
제품명: {product_name}

[타겟 페르소나]
특성: {persona_traits}
주요 관심사: {persona.get('value_focus', '제품 품질')}

[고객 리뷰 요약]
{review_text}

[핵심 포인트]
{highlights_text}

작성 규칙:
1. 반드시 다음 형식을 따르세요:

[제목]
(간결하고 임팩트 있게, 30~40자)

[본문]
(페르소나 공감과 제품 효과 중심, 200~300자)

2. 리뷰에서 확인 가능한 사실만 사용하세요.
3. 숫자, 할인율, 이벤트명은 절대 사용하지 마세요.
4. 페르소나의 가치관을 반영하되, 페르소나 이름(고객군명)은 절대 직접 언급하지 마세요.
5. 고객을 "당신", "이 제품을 원하는 분들" 등으로 표현하세요.
"""

        messages = [
            {"role": "system", "content": f"{persona.get('name', '고객')} 페르소나를 위한 마케팅 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        marketing_draft, duration = self.generate_text(messages, max_tokens=512, temperature=0.1)
        return marketing_draft, duration


# Ensure src directory is importable when running script directly
sys.path.insert(0, os.path.dirname(__file__))
from rag_utils import vectorize_texts, cosine, extract_candidate_texts, extract_highlight_snippet, build_persona_query


def generate_marketing_draft(persona, product, highlights):
    """템플릿 기반 마케팅 초안 생성"""
    title = f"{product.get('brand_name','')} {product.get('name','')} — {persona.get('name')}용 추천"
    intro = f"{persona.get('name')} 페르소나에 맞춘 제안: {persona.get('growth_point','')}."
    pain = f"페르소나 주요 고민: {persona.get('skin_type','')}. 가치 포인트: {persona.get('value_focus','')}."
    bullets = []
    for h in highlights:
        bullets.append('- ' + h)
    usage = '간단 사용 팁: 아침/저녁 스킨케어 마지막 단계에서 소량을 펴발라 흡수시켜 주세요.'
    cta = '지금 바로 만나보세요 — 한정 혜택을 확인하세요.'
    draft = '\n\n'.join([title, intro, pain, '\n'.join(bullets), usage, cta])
    return draft



def sanitize_filename(s):
    return re.sub(r'[^0-9a-zA-Z_-]', '_', s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persona', required=True, help='페르소나 인덱스(0~)또는 이름')
    parser.add_argument('--brand', required=True, help='브랜드명')
    parser.add_argument('--product', required=True, help='제품명(부분일치 허용)')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K 리뷰 수')
    parser.add_argument('--personas_path', default=None, help='페르소나 JSON 경로(테스트용)')
    parser.add_argument('--products_path', default=None, help='제품 JSON 경로(테스트용)')
    parser.add_argument('--use_local_model', action='store_true', help='로컬 Qwen 모델 사용 (마케팅 초안 생성)')
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-1.5B-Instruct', help='로컬 모델 ID')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    personas_path = args.personas_path or os.path.join(base, 'data', 'personas.json')
    products_path = args.products_path or os.path.join(base, 'data', 'products.json')
    personas = load_json(personas_path)
    products = load_json(products_path)

    persona = find_persona(personas, args.persona)
    product = find_product(products, args.brand, args.product)

    query = build_persona_query(persona)
    candidates = extract_candidate_texts(product)
    all_texts = [query] + candidates
    vectors = vectorize_texts(all_texts)
    q_vec = vectors[0]
    cand_vecs = vectors[1:]

    scores = [(cosine(q_vec, v), i, candidates[i]) for i,v in enumerate(cand_vecs)]
    scores.sort(reverse=True, key=lambda x: x[0])
    top = scores[:args.top_k]

    top_k_list = []
    highlights = []
    for score, idx, text in top:
        snippet = extract_highlight_snippet(text)
        top_k_list.append({'index': idx, 'score': float(score), 'text': text, 'snippet': snippet})
        highlights.append(snippet)

    draft = generate_marketing_draft(persona, product, highlights)

    product_basic = {
        'product_id': product.get('product_id'),
        'name': product.get('name'),
        'brand_name': product.get('brand_name'),
        'price': product.get('price'),
        'url': product.get('url')
    }

    llm_raw = None
    llm_parsed = None
    llm_error = None
    generation_time = None

    if args.use_local_model:
        try:
            print("\n[로컬 Qwen 모델 - 마케팅 초안 생성]")
            generator = LocalQwenGenerator(model_name=args.model_name)
            
            # Generate marketing draft
            print("[생성] 마케팅 초안을 생성 중...")
            llm_raw, generation_time = generator.generate_marketing_draft(
                product.get('brand_name', ''),
                product.get('name', ''),
                persona,
                product.get('reviews', []),
                highlights
            )
            print(f"[완료] ({generation_time:.2f}초)\n")
            
            llm_parsed = {
                'marketing_draft': llm_raw,
                'generation_time': generation_time
            }
        except Exception as e:
            llm_error = str(e)
            print(f"[오류] {llm_error}\n")

    out = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'persona_input': args.persona,
        'brand': args.brand,
        'product_query': args.product,
        'persona_profile': persona,
        'product_basic': product_basic,
        'top_k': top_k_list,
        'marketing_draft': draft
    }
    if llm_parsed is not None:
        out['marketing_draft_llm'] = llm_parsed
    if llm_raw is not None:
        out['llm_raw'] = llm_raw
    if llm_error is not None:
        out['llm_error'] = llm_error
    if generation_time is not None:
        out['generation_time_seconds'] = generation_time

    out_dir = os.path.join(base, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    fname = f"marketing_{sanitize_filename(product.get('brand_name',''))}_{sanitize_filename(product.get('product_id',''))}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print('✓ 저장 완료:', out_path)
    print('Top-K snippets:')
    for i,t in enumerate(top_k_list,1):
        print(f"  {i}. score: {round(t['score'],3)} | {t['snippet'][:100]}")
    
    if llm_raw:
        print('\n[LLM 생성 마케팅 초안]')
        print(llm_raw[:400])


if __name__ == '__main__':
    main()
