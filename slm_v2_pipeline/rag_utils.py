import re
import math
from collections import Counter
import torch
from sentence_transformers import SentenceTransformer, util

# 전역 임베딩 모델 (한 번만 로드)
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading SentenceTransformer model (jhgan/ko-sroberta-multitask)...")
        _embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return _embedder


def tokenize(text):
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w가-힣]+", ' ', text)
    toks = [t for t in text.split() if len(t) > 1]
    return toks


def vectorize_texts(texts):
    """SentenceTransformer를 사용한 의미 기반 벡터화"""
    if not texts:
        return []
    embedder = get_embedder()
    # 벡터를 numpy 배열로 반환 (cosine 함수에서 사용)
    vectors = embedder.encode(texts, convert_to_tensor=False)
    return vectors


def cosine(a, b):
    """SentenceTransformer 벡터를 위한 코사인 유사도"""
    if a is None or b is None:
        return 0.0
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


POSITIVE_KEYWORDS = [
    '좋', '만족', '추천', '재구매', '인생템', '효과', '흡수', '보습', '탄력', '광채', '진정', '재구매', '신뢰', '가볍', '리뉴얼'
]

HIGHLIGHT_KEYS = ['효과', '성분', '제형', '흡수', '보습', '재구매', '신뢰', '사용감', '탄력', '주름', '진정', '광채']


def is_positive_review(review):
    if not isinstance(review, dict):
        return False
    rating = review.get('rating')
    if rating:
        try:
            if int(rating) >= 4:
                return True
        except Exception:
            pass
    text = review.get('text','')
    for k in POSITIVE_KEYWORDS:
        if k in text:
            return True
    return False


def extract_candidate_texts(product):
    """제품의 긍정적인 리뷰 텍스트만 추출 (제품명, 카테고리 제외)"""
    texts = []
    
    # 긍정 리뷰 추출
    for r in product.get('reviews', []):
        if is_positive_review(r):
            text = r.get('text', '').strip()
            # 최소 길이 필터 (20자 이상만 포함)
            if len(text) > 20:
                texts.append(text)
    
    # 긍정 리뷰가 부족하면 모든 리뷰 추가
    if len(texts) < 3:
        for r in product.get('reviews', []):
            text = r.get('text', '').strip()
            if len(text) > 20 and text not in texts:
                texts.append(text)
    
    return texts


def extract_highlight_snippet(text):
    if not text:
        return ''
    sents = re.split(r'[\.\n]+', text)
    for s in sents:
        for k in HIGHLIGHT_KEYS:
            if k in s:
                return s.strip()
    t = text.strip()
    return (t[:200] + '...') if len(t) > 200 else t


def build_persona_query(persona):
    parts = []
    if persona.get('skin_type'):
        parts.append('주요 고민: ' + persona['skin_type'])
    if persona.get('value_focus'):
        parts.append('가치관: ' + persona['value_focus'])
    if persona.get('shopping_style'):
        parts.append('쇼핑스타일: ' + persona['shopping_style'])
    if persona.get('growth_point'):
        parts.append('성장 포인트: ' + persona['growth_point'])
    query = ' | '.join(parts)
    return query
