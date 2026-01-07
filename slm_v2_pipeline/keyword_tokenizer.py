"""
키워드 빈도 추출 유틸리티

리뷰를 토크나이징해서 빈도가 높은 키워드를 우선순위로 정렬
"""

import re
from collections import Counter


# 불용어 (제외할 단어들)
STOPWORDS = {
    # 일반적인 불용어
    '있다', '없다', '하다', '되다', '이다', '같다', '그렇다', '아니다',
    '것', '수', '등', '때', '년', '월', '일', '번', '개', '분',
    '저', '제', '이', '그', '저희', '우리', '너무', '정말', '진짜', '완전',
    '좀', '더', '많이', '잘', '또', '그냥', '아주', '매우', '조금',
    '처음', '항상', '계속', '다시', '바로', '먼저', '아직', '이미',
    '그래서', '그런데', '그러나', '하지만', '그리고',
    # 화장품 리뷰 공통 표현
    '사용', '제품', '구매', '가격', '배송', '리뷰', '추천', '효과',
    '좋아요', '좋습니다', '좋네요', '좋았어요', '좋아서',
    '샀어요', '샀는데', '받았어요', '왔어요',
    '아모레', '아모레몰', '아모레퍼시픽',
}

# 긍정 감성 키워드 (우선순위 부여)
POSITIVE_BOOST = {
    '촉촉', '보습', '수분', '탄력', '광채', '윤기', '부드러움', '산뜻',
    '순한', '저자극', '진정', '개선', '효과', '흡수', '가벼운', '편안',
    '향', '발림성', '지속력', '커버력', '밀착', '자연스러운',
    '가성비', '재구매', '만족', '최고', '인생템', '강추',
}

# 부정 키워드 (제외)
NEGATIVE_WORDS = {
    '아쉽', '별로', '싫', '불편', '끈적', '무거', '자극', '트러블',
    '비싸', '가격이', '배송이', '늦', '느리', '부족', '없어',
}


def tokenize_korean(text: str) -> list:
    """
    간단한 한국어 토크나이저
    형태소 분석기 없이 공백/기호 기준 분리 + 정규화
    """
    # 특수문자 제거 (이모지, 숫자 등)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # 공백 기준 분리
    tokens = text.split()
    
    # 2글자 이상만 유지
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 2]
    
    return tokens


def extract_keywords_by_frequency(reviews: list, top_k: int = 15) -> list:
    """
    리뷰 목록에서 빈도 높은 키워드 추출
    
    Args:
        reviews: 리뷰 텍스트 리스트
        top_k: 반환할 키워드 수
        
    Returns:
        [(keyword, count), ...] 빈도순 정렬
    """
    all_tokens = []
    
    for review in reviews:
        if isinstance(review, dict):
            text = review.get('text', '') or review.get('snippet', '')
        else:
            text = str(review)
        
        tokens = tokenize_korean(text)
        all_tokens.extend(tokens)
    
    # 불용어 제거 & 부정 키워드 제거
    filtered = [
        t for t in all_tokens 
        if t not in STOPWORDS and not any(neg in t for neg in NEGATIVE_WORDS)
    ]
    
    # 빈도 카운트
    counter = Counter(filtered)
    
    # 긍정 키워드 부스트 (빈도 x 1.5)
    for word, count in list(counter.items()):
        if any(pos in word for pos in POSITIVE_BOOST):
            counter[word] = int(count * 1.5)
    
    # 상위 k개 반환
    return counter.most_common(top_k)


def format_frequency_keywords(freq_keywords: list) -> str:
    """
    빈도순 키워드를 문자열로 포맷팅
    
    Args:
        freq_keywords: [(keyword, count), ...]
        
    Returns:
        "키워드1, 키워드2, 키워드3, ..."
    """
    return ', '.join([kw for kw, _ in freq_keywords])


def preprocess_reviews_with_frequency(reviews: list, top_k: int = 10) -> str:
    """
    리뷰를 토크나이징하고 빈도 높은 키워드를 우선 정렬해서 반환
    
    사용법:
        keywords = preprocess_reviews_with_frequency(product['reviews'])
        # "촉촉, 보습, 흡수, 순한, ..."
    """
    freq_keywords = extract_keywords_by_frequency(reviews, top_k)
    return format_frequency_keywords(freq_keywords)
