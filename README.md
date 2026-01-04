# 🌺 AmoRe CRM Generator

이 프로젝트는 **브랜드 톤을 유지한 CRM 메시지**를 자동으로 생성하는 파이프라인입니다.  
범용 AI의 획일화, 데이터 유출 리스크, 과도한 개인화 피로감을 줄이기 위해 **Qwen → EXAONE 이원 구조**로 설계되었습니다.

---

## 1. 프로젝트 배경과 필요성

CRM 메시지는 구매 전환, 재구매, 추천 행동을 직접 좌우하는 핵심 접점입니다.  
특히 뷰티/소비재 분야는 **브랜드 톤**이 곧 신뢰를 의미하기 때문에 문장 하나의 뉘앙스가 성과에 영향을 줍니다.

하지만 현실에서는 다음 문제가 반복적으로 발생합니다.

- **수작업 제작 부담**: 캠페인마다 수십 개의 페르소나/단계/스타일 조합을 수작업으로 작성해야 함
- **브랜드 톤 통제 어려움**: 범용 LLM은 톤이 획일화되거나 브랜드별 차이가 사라짐
- **정보 리스크**: 외부 API 사용 시 데이터 유출 우려, 생성 결과의 할루시네이션 위험
- **과도한 개인화 피로감**: 지나친 개인화 문구는 오히려 거부감을 유발

이 프로젝트는 위 문제를 해결하기 위해 **로컬 기반 Private AI + 구조화된 의사결정 체계**로 설계되었습니다.

---

## 2. 이 프로젝트가 해결하는 문제

- **브랜드 고유성 보존**: 한 가지 문체로 섞이지 않도록 각 브랜드의 톤앤매너를 반영
- **정보 정확성 확보**: 제품/리뷰 기반 사실 위주로 구성하고 할루시네이션을 억제
- **현업 의사결정 구조 반영**: 마케터가 실제로 사용하는 Why-Who-What 흐름을 그대로 반영
- **데이터 보안 강화**: 외부 API 의존도를 줄이고 로컬 추론 중심으로 구성
- **초개인화 피로감 완화**: 페르소나 기반의 구조적 개인화로 과도한 맞춤 표현을 회피

---

## 3. 해결 전략 요약

1) **Private AI 로컬 추론**  
   고객/브랜드 데이터를 외부로 전송하지 않고 로컬 환경에서 처리

2) **Brand Tone Foundation**  
   공식 브랜드 문구를 임베딩해 브랜드 고유 언어를 보존

3) **Why-Who-What 의사결정 구조**  
   마케터의 실제 사고 구조(AARRR + 페르소나 + 제품 컨셉)를 그대로 반영

4) **Qwen → EXAONE 이원 구조**  
   - Qwen: 사실 기반 초안 생성  
   - EXAONE: 브랜드 톤/CRM 설득 구조 보정

5) **RAG 기반 근거 확보**  
   리뷰/제품 정보를 검색해 사실성 기반 메시지 생성

---

## 4. 전체 흐름 한눈에 보기

1) **입력 파라미터 수집**  
   페르소나, 브랜드, 제품, 고객 단계(AARRR), 스타일, 이벤트 여부 등

2) **RAG 기반 근거 수집**  
   제품/리뷰/브랜드 정보를 검색해 요약에 필요한 근거를 확보

3) **Qwen 초안 생성**  
   사실 기반의 구조화된 초안(요약/핵심 포인트)을 생성

4) **EXAONE 톤 보정**  
   브랜드 톤과 CRM 전략을 반영해 최종 메시지로 다듬기

5) **JSON 결과 반환**  
   메시지 + 참고 정보 + 메타데이터를 구조화된 형태로 반환

---

## 5. 핵심 설계 원칙

### 3.1 브랜드 톤(Brand Tone Foundation)
- 브랜드 소개 문구, 공식 마케팅 텍스트를 임베딩해 **브랜드 고유 언어를 유지**

### 3.2 마케터 의사결정 구조
- **Why**: AARRR 단계로 발송 목적 설정  
- **Who**: 5대 페르소나로 타겟 세분화  
- **What**: 제품 DB 기반으로 메시지 컨셉 구성

### 3.3 이원 모델 구조
- **Qwen**: 사실 기반 초안 생성에 집중  
- **EXAONE**: 톤/문체/설득 구조를 정교화

---

## 6. 입력 파라미터 요약

- `persona`: 0~4 (페르소나 ID)
  - 0 Luxury Lover / 1 Sensitive Skin / 2 Budget Seeker / 3 Trend Follower / 4 Natural Beauty
- `brand`: 브랜드명 문자열
- `product`: 제품명 문자열
- `stage_index`: 0~4 (AARRR 단계)
  - 0 Acquisition / 1 Activation / 2 Retention / 3 Revenue / 4 Referral
- `style_index`: 0~5 (스타일 템플릿 인덱스)
- `is_event`: 0/1 (이벤트 여부)
- `top_k`: RAG 상위 후보 수 (기본값 3)

---

## 7. 실행 방법

### 5.1 설치
```bash
pip install -r AmoRe_crm_generator/requirements.txt
```

### 5.2 단일 실행 (CLI)
```bash
python AmoRe_crm_generator/src/run_qwen_exaone_pipeline.py \
  --persona 0 \
  --brand "아이오페" \
  --product "수분가득 콜라겐 크림 75ml" \
  --stage_index 1 \
  --style_index 2 \
  --is_event 1
```

### 5.3 배치 실행
`batch.json`
```json
[
  {"persona": 0, "brand": "아이오페", "product": "수분가득 콜라겐 크림 75ml", "stage_index": 1, "style_index": 2, "is_event": 1},
  {"persona": 2, "brand": "에뛰드", "product": "수분 크림", "stage_index": 0, "style_index": 0, "is_event": 0}
]
```
```bash
python AmoRe_crm_generator/src/run_qwen_exaone_pipeline.py \
  --batch_json batch.json \
  --out_path out.json
```

---

## 8. 서버/프론트 실행

### 6.1 FastAPI + ngrok (콜랩 등 외부 접근)
```bash
python AmoRe_crm_generator/server.py
```
- 실행 로그에 `ngrok tunnel: https://...` URL이 출력됩니다.
- 해당 URL로 접속하면 프론트 UI가 열립니다.

### 6.2 로컬 FastAPI
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
- 접속: `http://localhost:8000/`

### 6.3 로컬 CLI 사용도 가능
- 파이프라인은 **서버 없이도** CLI로 직접 실행 가능합니다.

---

## 9. 반환 결과 예시 (요약)

```json
{
  "qwen": {"draft": "..."},
  "exaone": {"result_raw": "..."},
  "stage_kr": "Retention",
  "objective": "...",
  "target_state": "...",
  "style_templates": ["..."],
  "selected_event": null,
  "timing": {"load": 0.2, "rag": 0.5, "qwen": 1.2, "exaone": 2.8, "total": 4.9}
}
```

---

## 10. 성능 로그 / 캐시

- `[Timing]` 로그: load / rag / qwen / exaone / total 시간 출력
- `[TimingAvg]` 로그: 100건 누적 평균 출력
- `--disable_cache`: 캐시 비활성화 (재현성 테스트용)

---

## 11. 파인튜닝 데이터/평가

- `finetuning/`에는 DPO 데이터 생성 및 평가용 코드가 있습니다.
- `finetuning/base vs. adapter comparison report.md` 에는 Base 모델과 LoRA 모델의 비교 레포트가 있습니다.
- 해당 레포트에서는 다양한 예시를 통해 GPT 평가/비GPT 지표를 함께 사용해 메시지 품질을 정량화합니다.
- DPO 파인튜닝용 데이터셋은 [![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Jinhyeok33/crm-dpo-dataset)
 에 있습니다.
- 파인튜닝된 LoRA 어댑터는 [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/Jinhyeok33/crm-dpo-adapter)
 에 있습니다.

---

## 12. 참고 문서

- 상세 설계 문서: `AmoRe_crm_generator/CRM AI Agent Proposal.md`
- 파인튜닝 비교 문서: `AmoRe_crm_generator/finetuning/base vs. adapter comparison report.md`
- 파이프라인 코드: `AmoRe_crm_generator/src/run_qwen_exaone_pipeline.py`
- 서버 코드: `AmoRe_crm_generator/server.py`
- 서버 참고 가이드: `AmoRe_crm_generator/frontend/README.md`

---

## 13. 주의사항

- 모델 추론은 T4 GPU 이상 사용을 권장합니다.
