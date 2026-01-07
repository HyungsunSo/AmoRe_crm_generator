"""
Sequential CRM Pipeline v2 - SLM-Optimized (HICR)

핵심 개선:
1. XML 태그 기반 구조화된 출력
2. 스타일 요소 주입 (예시 대신 수식어/어미/이모지 직접 제공)
3. EmotionPrompt 적용 (고위험 표현으로 집중도 향상)
4. 엄격한 금기어/허용어 명시
"""

import re
import json
import os

# 독립 실행 지원을 위한 import
try:
    from .model import get_creator, get_validator, get_polisher, get_styler
except ImportError:
    from model import get_creator, get_validator, get_polisher, get_styler


def load_brand_style(brand_name: str) -> dict:
    """Load style_elements from instagram_ground_truth.json"""
    # slm_v2_pipeline/steps_v2.py -> slm_v2_pipeline -> root
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt_path = os.path.join(base, 'data', 'instagram_ground_truth.json')
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt = json.load(f)
        return gt.get(brand_name, {}).get('style_elements', {})
    except:
        return {}


class BaseStep:
    def __init__(self, creator=None, validator=None):
        self.creator = creator if creator else get_creator()
        self.validator = validator if validator else get_validator()
        self.name = self.__class__.__name__

    def _clean_output(self, text: str) -> str:
        if not text: return ""
        # <think>...</think> 제거
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        return text.strip()

    def _extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content between XML tags"""
        pattern = rf'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()


# ============================================================
# Step 0: ReviewSummarizer (리뷰 → 긍정 키워드 추출)
# ============================================================
class ReviewSummarizer(BaseStep):
    """리뷰에서 긍정적인 키워드만 추출 (전처리)"""

    def run(self, reviews: str) -> tuple:
        prompt = f"""<TASK>
아래 리뷰에서 긍정적인 키워드만 추출하세요.
부정적인 내용은 무시하세요.
문장으로 쓰지 말고 키워드만 나열하세요.
</TASK>

<INPUT>
{reviews}
</INPUT>

<CONSTRAINTS>
- 긍정적인 내용만 추출
- 효과, 느낌, 장점 위주
- 부정적 키워드 제외
- 최대 10개 키워드
</CONSTRAINTS>

<OUTPUT_FORMAT>
<positive_keywords>
(쉼표로 구분된 긍정 키워드들)
</positive_keywords>
</OUTPUT_FORMAT>

키워드만 출력하세요."""

        result, dur = self.creator.generate([{"role": "user", "content": prompt}], max_tokens=2000)
        clean = self._clean_output(result)
        keywords = self._extract_xml_content(clean, 'positive_keywords')
        return keywords, dur


# Step 1: BriefGenerator (키워드 추출)
# ============================================================
class BriefGenerator(BaseStep):
    """제품 정보에서 핵심 키워드 추출 (문장 X, 키워드 나열 O)"""

    def run(self, product_name: str, highlights: str) -> tuple:
        prompt = f"""<TASK>
아래 리뷰 키워드를 정리해서 제품 브리프 키워드를 만드세요.
</TASK>

<제품명>
{product_name}
</제품명>

<리뷰_키워드>
{highlights}
</리뷰_키워드>

<OUTPUT_FORMAT>
<brief>
{product_name}, 효능키워드1, 효능키워드2, 장점키워드1, 장점키워드2
</brief>
</OUTPUT_FORMAT>

예시: "콜라겐 토너, 촉촉함, 탄력, 순한, 가성비"
주의: 오직 쉼표로 구분된 키워드만 출력. 문장 금지."""

        result, dur = self.creator.generate([{"role": "user", "content": prompt}], max_tokens=2000)
        clean = self._clean_output(result)
        brief = self._extract_xml_content(clean, 'brief')
        return brief, dur


# ============================================================
# Step 2: PersonaWriter (키워드에 감정 추가)
# ============================================================
class PersonaWriter(BaseStep):
    """키워드에 페르소나 맞춤 감정 키워드 추가"""

    PERSONA_EMOTION = {
        "Luxury_Lover": {"감정": "프리미엄, 고급스러운, 럭셔리"},
        "Sensitive_Skin": {"감정": "순한, 저자극, 안심"},
        "Budget_Seeker": {"감정": "가성비, 실속, 혜택"},
        "Trend_Follower": {"감정": "트렌디, 핫한, 인기"},
        "Natural_Beauty": {"감정": "자연스러운, 건강한, 순수한"},
        "default": {"감정": "좋은, 추천"},
    }

    def run(self, brief: str, persona_name: str) -> tuple:
        emotion_data = self.PERSONA_EMOTION.get(persona_name, self.PERSONA_EMOTION["default"])
        emotion_kw = emotion_data["감정"]

        # 예시 기반 프롬프트
        prompt = f"""다음 키워드들을 합쳐서 출력하세요.

키워드1: {brief}
키워드2: {emotion_kw}

예시 출력: 촉촉함, 탄력, 순한, 프리미엄, 고급스러운

출력:"""

        result, dur = self.creator.generate([{"role": "user", "content": prompt}], max_tokens=2000)
        clean = self._clean_output(result)
        # "출력:" 이후 내용만 추출
        if '출력' in clean:
            clean = clean.split('출력')[-1].strip()
        if ':' in clean and len(clean.split(':')[0]) < 10:
            clean = ':'.join(clean.split(':')[1:]).strip()
        return clean, dur


# ============================================================
# Step 3: GoalSetter (키워드에 CTA 키워드 추가)
# ============================================================
class GoalSetter(BaseStep):
    """키워드에 AARRR 스테이지 맞춤 CTA 키워드 추가"""

    CTA_KEYWORDS = {
        "Acquisition": "첫만남, 신규혜택, 지금시작",
        "Activation": "오늘시작, 체험, 도전",
        "Retention": "다시만나요, 리뉴얼, 감사",
        "Revenue": "특별혜택, 한정, 서두르세요",
        "Referral": "친구추천, 함께, 공유",
    }

    def run(self, keywords: str, stage: str) -> tuple:
        cta = self.CTA_KEYWORDS.get(stage, "지금확인")

        # 예시 기반 프롬프트
        prompt = f"""다음 키워드들을 합쳐서 출력하세요.

키워드1: {keywords}
키워드2: {cta}

예시 출력: 촉촉함, 탄력, 순한, 첫만남, 지금시작

출력:"""

        result, dur = self.creator.generate([{"role": "user", "content": prompt}], max_tokens=2000)
        clean = self._clean_output(result)
        # "출력:" 이후 내용만 추출
        if '출력' in clean:
            clean = clean.split('출력')[-1].strip()
        if ':' in clean and len(clean.split(':')[0]) < 10:
            clean = ':'.join(clean.split(':')[1:]).strip()
        return clean, dur


# ============================================================
# Step 4: BrandStyler (키워드를 문장으로 조합 - Qwen 4B 사용)
# ============================================================
class BrandStyler(BaseStep):
    """키워드들을 브랜드 스타일에 맞는 문장으로 조합 (Qwen 4B)"""

    def __init__(self):
        super().__init__()
        self.styler = get_styler()  # Qwen 4B 사용

    def run(self, keywords: str, brand_name: str) -> tuple:
        style = load_brand_style(brand_name)

        modifiers = ', '.join(style.get('modifiers', ['친근한', '자연스러운']))
        endings = ', '.join(style.get('endings', ['~해요', '~이에요']))
        emojis = ' '.join(style.get('emojis', ['🌿']))
        banned = ', '.join(style.get('banned_expressions', []))

        # /no_think로 thinking 모드 비활성화
        prompt = f"""/no_think
{brand_name} 브랜드 CRM 메시지를 작성하세요.

키워드: {keywords}

규칙:
- 어미: {endings} 중 하나
- 이모지: {emojis} 중 하나
- 분위기: {modifiers}
- 50자 내외
- 1~2문장

메시지:"""

        result, dur = self.styler.generate([{"role": "user", "content": prompt}], max_tokens=2000)
        clean = self._clean_output(result)
        # 결과에서 prefix 제거
        if ':' in clean and len(clean.split(':')[0]) < 15:
            clean = ':'.join(clean.split(':')[1:]).strip()
        return clean, dur


# ============================================================
# Step 5: TitleGenerator (본문 기반 제목 생성)
# ============================================================
class TitleGenerator(BaseStep):
    """완성된 본문을 기반으로 CRM 메시지 제목 생성 (Qwen)"""

    def run(self, body_text: str, brand_name: str) -> tuple:
        prompt = f"""<TASK>
당신은 {brand_name}의 CRM 마케팅 전문가입니다.
아래 본문을 읽고, 클릭을 유도하는 매력적인 제목을 작성하세요.
</TASK>

<INPUT>
본문: {body_text}
</INPUT>

<CONSTRAINTS>
- 반드시 한국어로 작성
- 15~25자 이내
- 본문의 핵심 혜택/감정을 압축
- 이모지 1개 사용 가능
- 호기심/긴박감 유발
</CONSTRAINTS>

<OUTPUT_FORMAT>
<title>
(제목만 출력)
</title>
</OUTPUT_FORMAT>"""

        result, dur = self.creator.generate([{"role": "user", "content": prompt}], max_tokens=2000)
        clean = self._clean_output(result)
        title = self._extract_xml_content(clean, 'title')
        return title, dur


# ============================================================
# Step 6: FinalPolisher (최종 윤문 - HyperCLOVAX 사용)
# ============================================================
class FinalPolisher(BaseStep):
    """번역투 제거 및 자연스러운 한국어로 최종 윤문 (클로바 사용)"""

    def __init__(self):
        super().__init__()
        self.polisher = get_polisher()  # 클로바 모델 사용

    def run(self, text: str) -> tuple:
        # 클로바는 XML 파싱을 못하므로 단순한 형식 사용
        prompt = f"""다음 CRM 메시지를 더 자연스럽고 매력적인 한국어로 다듬어주세요.

원본: {text}

수정 규칙:
- 어색한 번역투 표현을 자연스럽게 수정
- 의미는 유지하면서 더 부드럽게
- 50자 내외로 유지

결과만 출력하세요:"""

        result, dur = self.polisher.generate([{"role": "user", "content": prompt}], max_tokens=2000)
        clean = self._clean_output(result)
        # 결과에서 "결과:" 같은 prefix 제거
        if ':' in clean and len(clean.split(':')[0]) < 10:
            clean = ':'.join(clean.split(':')[1:]).strip()
        return clean, dur
