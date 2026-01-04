import json
import random
import csv

# 경로
PRODUCTS_JSON = "../data/products.json"
OUTPUT_CSV = "random_persona_campaign.csv"

# 로드
with open(PRODUCTS_JSON, "r", encoding="utf-8") as f:
    products = json.load(f)

rows = []

N = 2000  # 생성할 row 수 (원하면 조절)

for _ in range(N):
    p = random.choice(products)

    row = {
        "persona": random.randint(0, 4),
        "brand": p["brand_name"],
        "product": p["name"],
        "stage_index": random.randint(0, 4),
        "style_index": random.randint(0, 5),
        "is_event": random.randint(0, 1),
    }

    rows.append(row)

# CSV 저장
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "persona",
            "brand",
            "product",
            "stage_index",
            "style_index",
            "is_event",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV 생성 완료: {OUTPUT_CSV}")
