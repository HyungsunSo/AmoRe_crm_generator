import os
import json
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_DIR = r"C:\Project-CRM_Agent\CRM-Insta-Hera"
OUTPUT_JSON = r"C:\Project-CRM_Agent\insta-Hera_ocr_to_json.json"

results = []
processed = 0

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, filename)
    print(f"OCR 처리 중: {filename}")

    try:
        img = Image.open(img_path)
    except Exception as e:
        print("이미지 열기 실패:", filename, e)
        continue

    text = pytesseract.image_to_string(
        img,
        lang="kor+eng",
        config="--psm 6"
    )

    results.append({
        "file_name": filename,
        "raw_text": text.strip()
    })

    processed += 1

print("총 OCR 처리 이미지 수:", processed)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("JSON 파일 생성 완료:", OUTPUT_JSON)
