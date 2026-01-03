# ------------------------------------------
# Step 0. 원본 JSON 로딩
# ------------------------------------------

import pandas as pd
import re
from datetime import date

INPUT_JSON_PATH = "insta-Laneige_ocr_to_json.json"
OUTPUT_JSON_PATH = "insta-Laneige_json_processing.json"

df = pd.read_json(INPUT_JSON_PATH)


# ------------------------------------------
# Step 1. JSON 전처리 (caption, date 생성)
# ------------------------------------------

def preprocess_instagram_ocr(df, account_name="laneige_kr"):
    def process_text(text):
        if not isinstance(text, str):
            return None, "", None

        post_id = None

        # 1. 계정명 이전 텍스트 제거
        account_idx = text.find(account_name)
        if account_idx != -1:
            text = text[account_idx:]

        # 2. 마지막 줄을 날짜로 분리
        lines = text.strip().split("\n")
        date_text = lines[-1].strip() if len(lines) > 1 else None
        text_body = "\n".join(lines[:-1])

        # 3. 개행 정리
        text_body = re.sub(r"\n{2,}", " ", text_body)
        text_body = re.sub(r"\n", " ", text_body)
        text_body = re.sub(r"\s+", " ", text_body).strip()

        # 4. 계정명 분리 → id 컬럼
        if text_body.startswith(account_name):
            post_id = account_name
            text_body = text_body[len(account_name):].strip()

        return post_id, text_body, date_text

    df[["id", "caption", "date"]] = df["raw_text"].apply(
        lambda x: pd.Series(process_text(x))
    )

    return df



# ------------------------------------------
# Step 2. 날짜 정규화
# ------------------------------------------

def normalize_date(raw_date, year=2025):
    if not isinstance(raw_date, str):
        return None, "invalid"

    nums = list(map(int, re.findall(r"\d+", raw_date)))
    candidates = []

    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            month, day = nums[i], nums[j]
            if 1 <= month <= 12 and 1 <= day <= 31:
                try:
                    candidates.append(date(year, month, day))
                except ValueError:
                    pass

    if len(candidates) == 1:
        return candidates[0].isoformat(), "high"
    elif len(candidates) > 1:
        return candidates[0].isoformat(), "medium"
    else:
        return None, "low"


# ------------------------------------------
# Step 3. 해시태그 분리
# ------------------------------------------

def extract_hashtags(text):
    if not isinstance(text, str):
        return []
    return re.findall(r"#(\w+)", text)


# ------------------------------------------
# Step 4. 실행 파이프라인
# ------------------------------------------

df_processed = preprocess_instagram_ocr(df.copy())

df_processed[["normalized_date", "date_confidence"]] = (
    df_processed["date"]
    .apply(lambda x: pd.Series(normalize_date(x)))
)

df_processed["tag"] = df_processed["caption"].apply(extract_hashtags)


# ------------------------------------------
# Step 5. JSON 저장 (raw_text 제외)
# ------------------------------------------

df_processed.drop(columns=["raw_text"]).to_json(
    OUTPUT_JSON_PATH,
    orient="records",
    force_ascii=False,
    indent=2
)
