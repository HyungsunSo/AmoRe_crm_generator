# SLM v2 Pipeline vs Qwen-Exaone Pipeline 호환성 검증

## 입력 파라미터 비교

| 파라미터 | SLM v2 | Qwen-Exaone | 호환 |
|---------|--------|-------------|------|
| `--persona` | ✅ 필수 | ✅ 필수 | ✅ 동일 |
| `--brand` | ✅ 필수 | ✅ 필수 | ✅ 동일 |
| `--product` | ✅ 필수 | ✅ 필수 | ✅ 동일 |
| `--stage_index` | ✅ 필수 (0~4) | ✅ 필수 (0~4) | ✅ 동일 |
| `--top_k` | ✅ 선택 (default=3) | ✅ 선택 (default=5) | ⚠️ 기본값 다름 |
| `--out_dir` | ✅ 선택 | ❌ `--out_path` 사용 | ⚠️ 이름 다름 |
| `--qwen_model` | ❌ 없음 (내장) | ✅ 선택 | - |
| `--exa_model` | ❌ 없음 | ✅ 선택 | - |
| `--is_event` | ❌ 없음 | ✅ 선택 | - |
| `--style_index` | ❌ 없음 | ✅ 선택 | - |

## 공통 유틸리티 함수

| 함수 | SLM v2 | Qwen-Exaone |
|------|--------|-------------|
| `top_highlights_for_product()` | ✅ 사용 | ✅ 사용 |
| `find_persona()` | ✅ 사용 | ✅ 사용 |
| `find_product()` | ✅ 사용 | ✅ 사용 |
| `load_json()` | ✅ 사용 | ✅ 사용 |
| `build_persona_query()` | ✅ 사용 | ✅ 사용 |
| `extract_candidate_texts()` | ✅ 사용 | ✅ 사용 |
| `vectorize_texts()` | ✅ 사용 | ✅ 사용 |

## AARRR 스테이지 매핑

| Index | Stage | 두 파이프라인 모두 동일 |
|-------|-------|----------------------|
| 0 | Acquisition | ✅ |
| 1 | Activation | ✅ |
| 2 | Retention | ✅ |
| 3 | Revenue | ✅ |
| 4 | Referral | ✅ |

## 데이터 파일 의존성

| 파일 | SLM v2 | Qwen-Exaone |
|------|--------|-------------|
| `data/personas.json` | ✅ | ✅ |
| `data/products.json` | ✅ | ✅ |
| `data/instagram_ground_truth.json` | ✅ | ❌ (brand_stories.json) |

## 결론

✅ **핵심 입력 파라미터 완전 호환**
- `--persona`, `--brand`, `--product`, `--stage_index` 4개 필수 파라미터 동일
- 동일한 RAG 유틸리티 함수 사용
- 동일한 AARRR 스테이지 매핑

⚠️ **차이점**
1. `--top_k` 기본값 다름 (SLM v2: 3, Qwen-Exaone: 5)
2. 출력 경로 파라미터 이름 다름 (`--out_dir` vs `--out_path`)
3. SLM v2는 `brand_styles.json` 사용, Qwen-Exaone은 `brand_stories.json` 사용
4. Qwen-Exaone은 추가 옵션 지원 (`--is_event`, `--style_index`)
