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
from functools import lru_cache
from datetime import datetime, timezone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from rag_utils import vectorize_texts, cosine, extract_candidate_texts, extract_highlight_snippet, build_persona_query


@lru_cache(maxsize=None)
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



_PERSONA_INDEX_CACHE = {}
_PRODUCT_INDEX_CACHE = {}


def _get_persona_index(personas):
    key = id(personas)
    cached = _PERSONA_INDEX_CACHE.get(key)
    if cached and cached.get("length") == len(personas):
        return cached.get("index", {})

    index = {}
    for idx, persona in enumerate(personas):
        name = str(persona.get("name", "")).lower()
        if name:
            index[name] = persona
        index[str(idx)] = persona

    _PERSONA_INDEX_CACHE[key] = {"length": len(personas), "index": index}
    return index


def _get_product_index(products):
    key = id(products)
    cached = _PRODUCT_INDEX_CACHE.get(key)
    if cached and cached.get("length") == len(products):
        return cached

    exact = {}
    by_brand = {}
    for product in products:
        brand = (product.get("brand_name", "") or "").strip().lower()
        name = (product.get("name", "") or "").strip().lower()
        if brand and name:
            exact[(brand, name)] = product
        if brand:
            by_brand.setdefault(brand, []).append(product)

    cached = {"length": len(products), "exact": exact, "by_brand": by_brand}
    _PRODUCT_INDEX_CACHE[key] = cached
    return cached


def find_persona(personas, persona_input):
    # persona_input can be index or name
    try:
        idx = int(persona_input)
        if 0 <= idx < len(personas):
            return personas[idx]
    except Exception:
        idx = None

    index = _get_persona_index(personas)
    key = str(persona_input).lower()
    persona = index.get(key)
    if persona:
        return persona

    # lookup by name fallback
    for p in personas:
        if p.get("name", "").lower() == key:
            return p
    raise ValueError("페르소나를 찾을 수 없습니다: %s" % persona_input)


def find_product(products, brand, product_name):
    brand_l = (brand or "").strip().lower()
    name_l = (product_name or "").strip().lower()
    index = _get_product_index(products)
    exact = index.get("exact", {})
    by_brand = index.get("by_brand", {})

    exact_key = (brand_l, name_l)
    if exact_key in exact:
        return exact[exact_key]

    if brand_l in by_brand:
        for p in by_brand[brand_l]:
            if name_l in p.get("name", "").strip().lower():
                return p

    for p in products:
        if brand_l in p.get("brand_name", "").strip().lower() or name_l in p.get("name", "").strip().lower():
            return p
    raise ValueError("제품을 찾을 수 없습니다: %s / %s" % (brand, product_name))


def get_device():
    """Get appropriate device for model inference."""
    # MPS는 생성 작업에서 문제가 있을 수 있으므로 CUDA만 사용하고 나머지는 CPU 사용
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalQwenGenerator:
    """로컬 Qwen 모델을 사용하여 마케팅 초안을 생성합니다."""
    _CACHE = {}
    CACHE_ENABLED = True
    
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", use_cache=True):
        self.device = get_device()
        self.model_name = model_name
        cache_key = (self.device, model_name)
        cache_allowed = use_cache and self.CACHE_ENABLED
        cached = self._CACHE.get(cache_key) if cache_allowed else None
        if cached:
            self.tokenizer = cached["tokenizer"]
            self.model = cached["model"]
            print(f"[로컬 Qwen] 캐시 로딩: {model_name}")
            return

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
        if cache_allowed:
            self._CACHE[cache_key] = {
                "tokenizer": self.tokenizer,
                "model": self.model,
            }
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
    
    def generate_text_batch(self, messages_list, max_tokens=512, temperature=0.1):
        if not messages_list:
            return [], 0.0

        input_texts = []
        for messages in messages_list:
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                input_text = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in messages]
                )
            input_texts.append(input_text)

        t_start = time.time()
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

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
        t_end = time.time()

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        outputs = []
        for i, input_len in enumerate(input_lengths):
            generated_ids = output_ids[i][int(input_len):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs.append(generated_text.strip())
        return outputs, t_end - t_start

    def build_marketing_messages(
        self,
        brand_name,
        product_name,
        persona,
        reviews,
        highlights,
        campaign_event_info=None,
    ):
        persona_traits = ", ".join(persona.get("traits", []) if isinstance(persona.get("traits"), list) else [])
        review_text = "\n".join([f"- {r.get('text', '')[:150]}" for r in reviews[:3]])
        highlights_text = "\n".join([f"- {h}" for h in highlights[:3]])

        event_section = ""
        if campaign_event_info:
            event_section = f"""
[캠페인/이벤트 정보]
이벤트명: {campaign_event_info.get('name', '')}
상세 내용: {campaign_event_info.get('detail', '')}
"""

        prompt = f"""당신은 마케팅 카피라이터입니다. 아래 정보를 바탕으로 마케팅 초안을 작성하세요.

[제품]
브랜드: {brand_name}
제품명: {product_name}

[타겟 페르소나]
특성: {persona_traits}
주요 관심사: {persona.get('value_focus', '제품 품질')}
{event_section}
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
4. 단, [캠페인/이벤트 정보]가 제공된 경우 해당 내용은 적극 활용하세요
5. 페르소나의 가치관을 반영하되, 페르소나 이름(고객군명)은 절대 직접 언급하지 마세요.
6. 고객을 "당신", "이 제품을 원하는 분들" 등으로 표현하세요.
"""

        return [
            {"role": "system", "content": f"{persona.get('name', '고객')} 페르소나를 위한 마케팅 전문가입니다."},
            {"role": "user", "content": prompt},
        ]

    def generate_marketing_draft(self, brand_name, product_name, persona, reviews, highlights, campaign_event_info=None):
        """생성: 마케팅 초안 (One-Stage)."""
        messages = self.build_marketing_messages(
            brand_name,
            product_name,
            persona,
            reviews,
            highlights,
            campaign_event_info=campaign_event_info,
        )
        marketing_draft, duration = self.generate_text(messages, max_tokens=512, temperature=0.1)
        return marketing_draft, duration

    def generate_marketing_draft_batch(self, items, max_tokens=512, temperature=0.1):
        messages_list = []
        for item in items:
            messages_list.append(
                self.build_marketing_messages(
                    item["brand_name"],
                    item["product_name"],
                    item["persona"],
                    item.get("reviews", []),
                    item.get("highlights", []),
                    campaign_event_info=item.get("campaign_event_info"),
                )
            )
        drafts, duration = self.generate_text_batch(
            messages_list,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return drafts, duration


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
