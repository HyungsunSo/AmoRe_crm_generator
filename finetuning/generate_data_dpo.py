import argparse
import csv
import inspect
import json
import os
import re
import sys
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from io import StringIO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "random_persona_campaign.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "finetuning_data_dpo", "cycle_01.json")
SRC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "src"))

def _log(message):
    print(message)


def _import_pipeline_main():
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    try:
        from run_qwen_exaone_pipeline import main as pipeline_main
    except Exception as exc:
        raise ImportError(
            "Failed to import main from ../src/run_qwen_exaone_pipeline.py"
        ) from exc
    return pipeline_main


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _load_pairs(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            persona_raw = row.get("persona", "").strip()
            brand_raw = row.get("brand", "").strip()
            product_raw = row.get("product", "").strip()
            stage_raw = row.get("stage_index", "").strip()
            style_raw = row.get("style_index", "").strip()
            if not persona_raw or not brand_raw or not product_raw:
                continue
            if not stage_raw or not style_raw:
                continue
            try:
                persona = int(persona_raw)
                stage_index = int(stage_raw)
                style_index = int(style_raw)
            except ValueError:
                continue
            yield {
                "persona": persona,
                "brand": brand_raw,
                "product": product_raw,
                "stage_index": stage_index,
                "style_index": style_index,
                "is_event": _parse_bool(row.get("is_event", "")),
            }


def _format_prompt(summarization):
    if summarization is None:
        return ""
    if isinstance(summarization, str):
        return summarization.strip()
    return json.dumps(summarization, ensure_ascii=False, indent=2)


def _candidate_text(candidate):
    if isinstance(candidate, dict):
        return (
            candidate.get("text")
            or candidate.get("crm_message")
            or candidate.get("message")
            or candidate.get("content")
        )
    return str(candidate)


def _normalize_candidates(raw):
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "candidates" in raw:
            raw = raw["candidates"]
        elif "messages" in raw:
            raw = raw["messages"]
        elif "crm_messages" in raw:
            raw = raw["crm_messages"]
        elif "crm_message" in raw and isinstance(raw["crm_message"], list):
            raw = raw["crm_message"]
        else:
            raw = [raw]
    if not isinstance(raw, list):
        raw = [raw]

    normalized = []
    for idx, item in enumerate(raw):
        text = _candidate_text(item)
        if not text:
            continue
        normalized.append({"response_id": idx, "text": text})
    return normalized


def _dedupe_candidates(items):
    seen = set()
    result = []
    for item in items:
        text = _candidate_text(item)
        if not text:
            continue
        key = text.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _extract_response_text(data):
    if isinstance(data, dict):
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = data.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and isinstance(block.get("text"), str):
                            parts.append(block["text"])
                        elif isinstance(block, str):
                            parts.append(block)
                elif isinstance(content, str):
                    parts.append(content)
            if parts:
                return "".join(parts).strip()

    raise ValueError(f"Invalid evaluator response: {data}")


def _call_gpt(prompt_text, candidates):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    candidate_lines = []
    for idx, candidate in enumerate(candidates):
        candidate_lines.append(f"[{idx}] {candidate.get('text', '')}")
    candidate_block = "\n\n".join(candidate_lines)

    system_prompt = (
        "너는 마케팅 문장 평가자다.\n"
        "목표는 “전환 가능성이 더 높은 CRM 메시지”를 고르는 것이다.\n\n"
        "다음 기준으로 두 응답을 비교하라:\n"
        "1. 수신자가 실제 행동(클릭/재구매)을 할 가능성\n"
        "2. persona와 구매 단계 적합성\n"
        "3. 상품·브랜드 핵심 장점 전달력\n"
        "4. 불필요한 장식 없이 명확한가\n\n"
        "더 나은 쪽을 선택하라."
    )
    user_prompt = (
        "요약:\n"
        f"{prompt_text}\n\n"
        "후보:\n"
        f"{candidate_block}\n\n"
        "더 나은 후보의 인덱스만 정수로 반환하라."
    )

    payload = {
        "model": "gpt-5-nano",
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc

    content = _extract_response_text(data)
    match = re.search(r"-?\d+", content)
    if not match:
        raise ValueError(f"Invalid evaluator response: {content}")
    choice = int(match.group(0))
    if choice < 0 or choice >= len(candidates):
        raise ValueError(f"Evaluator index out of range: {choice}")
    return choice


def _load_existing_records(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def _save_records(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _extract_pipeline_output(result):
    if isinstance(result, dict):
        summarization = result.get("summarization")
        crm_message = result.get("crm_message")
    elif isinstance(result, (list, tuple)) and len(result) >= 2:
        summarization, crm_message = result[0], result[1]
    elif isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unexpected pipeline output: {result}") from exc
        return _extract_pipeline_output(parsed)
    else:
        raise ValueError(f"Unexpected pipeline output: {result}")
    return summarization, crm_message


def _parse_stdout_payload(stdout_text):
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError("Pipeline did not return a usable payload.")


def _run_pipeline_via_argv(pipeline_main, row):
    argv = [
        "run_qwen_exaone_pipeline.py",
        "--persona",
        str(row["persona"]),
        "--brand",
        row["brand"],
        "--product",
        row["product"],
        "--stage_index",
        str(row["stage_index"]),
        "--style_index",
        str(row["style_index"]),
        "--is_event",
        "1" if row.get("is_event", False) else "0",
    ]
    buf = StringIO()
    old_argv = sys.argv
    try:
        sys.argv = argv
        with redirect_stdout(buf):
            result = pipeline_main()
    finally:
        sys.argv = old_argv
    if result is not None:
        return result
    return _parse_stdout_payload(buf.getvalue())


def _run_pipeline(pipeline_main, row):
    params = {
        "persona": row["persona"],
        "brand": row["brand"],
        "product": row["product"],
        "stage_index": row["stage_index"],
        "style_index": row["style_index"],
        "is_event": row.get("is_event", False),
    }
    try:
        sig = inspect.signature(pipeline_main)
    except (TypeError, ValueError):
        sig = None

    if sig is not None and len(sig.parameters) == 0:
        return _run_pipeline_via_argv(pipeline_main, row)

    try:
        return pipeline_main(**params)
    except TypeError:
        pass

    try:
        if sig is not None and len(sig.parameters) == 1:
            return pipeline_main(row)
    except TypeError:
        pass

    ordered = [
        params["persona"],
        params["brand"],
        params["product"],
        params["stage_index"],
        params["style_index"],
        params["is_event"],
    ]
    return pipeline_main(*ordered)


def _collect_candidates(inference_pipeline, row, num_candidates):
    summarization = None
    candidates = []
    for attempt in range(1, num_candidates + 1):
        _log(f"  [Attempt {attempt}/{num_candidates}] Running pipeline")
        result = _run_pipeline(inference_pipeline, row)
        s, message = _extract_pipeline_output(result)
        if summarization is None and s:
            summarization = s
        if isinstance(message, str) and message.strip():
            candidates.append({"text": message.strip()})
    return summarization, candidates


def generate_dpo_data(csv_path, output_path, max_rows=None, num_candidates=4):
    inference_pipeline = _import_pipeline_main()
    records = _load_existing_records(output_path)
    _log(f"CSV: {csv_path}")
    _log(f"Output: {output_path}")
    _log(f"Candidates per row: {num_candidates}")
    _log(f"Loaded existing records: {len(records)}")

    added = 0
    for idx, row in enumerate(_load_pairs(csv_path)):
        if max_rows is not None and idx >= max_rows:
            break

        _log(
            "[Row {idx}] persona={persona} brand={brand} product={product} "
            "stage_index={stage_index} style_index={style_index} is_event={is_event}".format(
                idx=idx,
                persona=row["persona"],
                brand=row["brand"],
                product=row["product"],
                stage_index=row["stage_index"],
                style_index=row["style_index"],
                is_event=row.get("is_event", False),
            )
        )

        try:
            summarization, candidates = _collect_candidates(
                inference_pipeline, row, num_candidates
            )
        except Exception as exc:
            _log(f"[Row {idx}] Pipeline error: {exc}")
            continue

        prompt_text = _format_prompt(summarization)
        if not prompt_text:
            _log(f"[Row {idx}] Empty summarization, skipping")
            continue

        _log(f"[Row {idx}] Raw candidates: {len(candidates)}")
        candidates = _dedupe_candidates(_normalize_candidates(candidates))
        _log(f"[Row {idx}] Deduped candidates: {len(candidates)}")
        if len(candidates) < 2:
            _log(f"[Row {idx}] Not enough candidates, skipping")
            continue

        try:
            best_idx = _call_gpt(prompt_text, candidates)
        except Exception as exc:
            _log(f"[Row {idx}] Evaluator error: {exc}")
            continue

        _log(f"[Row {idx}] Best candidate index: {best_idx}")
        best_text = candidates[best_idx]["text"]
        for candidate in candidates:
            if candidate is candidates[best_idx]:
                continue
            rejected_text = candidate["text"]
            if not rejected_text:
                continue
            records.append(
                {
                    "prompt": prompt_text,
                    "chosen": best_text,
                    "rejected": rejected_text,
                }
            )
            added += 1

    _save_records(output_path, records)
    _log(f"Saved {added} new records to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=DEFAULT_CSV)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_candidates", type=int, default=4)
    args = parser.parse_args()

    generate_dpo_data(
        args.csv_path,
        args.output_path,
        args.max_rows,
        args.num_candidates,
    )


if __name__ == "__main__":
    main()
