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


def _format_event(selected_event):
    if selected_event in (None, "", {}):
        return "없음"
    if isinstance(selected_event, dict):
        for key in ("title", "name", "event_name", "event"):
            if selected_event.get(key):
                return str(selected_event.get(key))
        return json.dumps(selected_event, ensure_ascii=False)
    return str(selected_event)


def _format_price(price):
    if price in (None, ""):
        return ""
    if isinstance(price, (int, float)):
        return f"{int(price):,}원"
    text = str(price).strip()
    if not text:
        return ""
    if "원" in text:
        return text
    if text.replace(",", "").isdigit():
        return f"{int(text.replace(',', '')):,}원"
    return text


def _format_persona(persona_profile):
    if not isinstance(persona_profile, dict):
        return str(persona_profile or "")
    name = persona_profile.get("name", "")
    extras = []
    value_focus = persona_profile.get("value_focus")
    skin_type = persona_profile.get("skin_type")
    traits = persona_profile.get("traits")
    shopping_style = persona_profile.get("shopping_style")
    if value_focus:
        extras.append(str(value_focus))
    if skin_type:
        extras.append(str(skin_type))
    if traits:
        if isinstance(traits, list):
            extras.append(", ".join([str(t) for t in traits if t]))
        else:
            extras.append(str(traits))
    if shopping_style:
        extras.append(str(shopping_style))
    extra_text = ", ".join([e for e in extras if e])
    if name and extra_text:
        return f"{name} ({extra_text})"
    return name or extra_text


def _build_prompt_text(meta, fallback_text):
    persona = _format_persona(meta.get("persona_profile") if isinstance(meta, dict) else None)
    stage = ""
    if isinstance(meta, dict):
        stage = meta.get("stage_name") or meta.get("stage_kr") or ""
    brand = meta.get("brand") if isinstance(meta, dict) else ""
    product_basic = meta.get("product_basic") if isinstance(meta, dict) else None
    product_name = ""
    price = ""
    if isinstance(product_basic, dict):
        product_name = product_basic.get("name", "") or ""
        price = _format_price(product_basic.get("price"))
    product_query = meta.get("product_query") if isinstance(meta, dict) else ""
    if not product_name:
        product_name = product_query or ""

    event_text = _format_event(meta.get("selected_event") if isinstance(meta, dict) else None)

    lines = ["[컨텍스트]"]
    if persona:
        lines.append(f"- Persona: {persona}")
    if stage:
        lines.append(f"- Stage: {stage}")
    if brand or product_name:
        lines.append(f"- Brand/Product: {brand} / {product_name}".strip())
    if price:
        lines.append(f"- Price: {price}")
    lines.append(f"- Event: {event_text}")

    prompt = "\n".join(lines).strip()
    if prompt:
        return prompt
    return _format_prompt(fallback_text)


def _candidate_text(candidate):
    if isinstance(candidate, dict):
        return (
            candidate.get("text")
            or candidate.get("crm_message")
            or candidate.get("message")
            or candidate.get("content")
        )
    return str(candidate)


def _candidate_meta(candidate):
    if isinstance(candidate, dict):
        return candidate.get("meta") or {}
    return {}


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
        normalized.append(
            {
                "response_id": idx,
                "text": text,
                "meta": _candidate_meta(item),
            }
        )
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


def _format_meta(meta):
    if not isinstance(meta, dict):
        return str(meta)

    lines = []
    persona_profile = meta.get("persona_profile")
    if persona_profile is not None:
        lines.append(f"persona_profile: {json.dumps(persona_profile, ensure_ascii=False)}")
    brand = meta.get("brand")
    if brand:
        lines.append(f"brand: {brand}")
    stage_kr = meta.get("stage_kr")
    if stage_kr:
        lines.append(f"stage_kr: {stage_kr}")
    objective = meta.get("objective")
    if objective:
        lines.append(f"objective: {objective}")
    target_state = meta.get("target_state")
    if target_state:
        lines.append(f"target_state: {target_state}")
    style_templates = meta.get("style_templates")
    if style_templates:
        if isinstance(style_templates, list):
            lines.append("style_templates:")
            for item in style_templates:
                lines.append(f"- {item}")
        else:
            lines.append(f"style_templates: {style_templates}")
    selected_event = meta.get("selected_event")
    if selected_event is not None:
        if isinstance(selected_event, (dict, list)):
            event_text = json.dumps(selected_event, ensure_ascii=False)
        else:
            event_text = str(selected_event)
        lines.append(f"selected_event: {event_text}")

    return "\n".join(lines) if lines else "(context unavailable)"


def _call_gpt(prompt_text, candidates):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    candidate_lines = []
    for idx, candidate in enumerate(candidates):
        meta_block = _format_meta(candidate.get("meta", {}))
        candidate_lines.append(
            f"[{idx}]\n"
            f"crm_message:\n{candidate.get('text', '')}\n\n"
            f"context:\n{meta_block}"
        )
    candidate_block = "\n\n".join(candidate_lines)

    system_prompt = (
        "너는 CRM 메시지 평가자다.\n"
        "목표는 전환 가능성이 더 높은 메시지를 고르는 것이다.\n"
        "각 후보에는 메시지와 그 메시지에 사용된 컨텍스트가 함께 주어진다.\n\n"
        "다음 기준으로 비교하라:\n"
        "1. 수신자가 실제 행동(클릭/재구매)을 할 가능성\n"
        "2. persona 및 stage_kr/objective/target_state 적합성\n"
        "3. 브랜드 핵심 장점 전달력\n"
        "4. style_templates 및 selected_event 반영 정도\n"
        "5. 불필요한 장식 없이 명확한가\n\n"
        "더 나은 후보 하나를 선택하라."
    )
    user_prompt = (
        "컨텍스트:\n"
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
        if "qwen" in result or "exaone" in result:
            qwen = result.get("qwen", {}) or {}
            exaone = result.get("exaone", {}) or {}
            summarization = qwen.get("draft") or qwen.get("qwen_draft")
            crm_message = exaone.get("result_raw") or exaone.get("crm_message")
            meta = {
                "persona_profile": result.get("persona_profile"),
                "brand": result.get("brand"),
                "product_basic": result.get("product_basic"),
                "product_query": result.get("product_query"),
                "stage_name": result.get("stage_name"),
                "stage_kr": result.get("stage_kr"),
                "objective": result.get("objective"),
                "target_state": result.get("target_state"),
                "style_templates": result.get("style_templates"),
                "selected_event": result.get("selected_event"),
            }
            return summarization, crm_message, meta
        summarization = result.get("summarization")
        crm_message = result.get("crm_message")
        return summarization, crm_message, {}
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        summarization, crm_message = result[0], result[1]
        return summarization, crm_message, {}
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Unexpected pipeline output: {result}") from exc
        return _extract_pipeline_output(parsed)
    raise ValueError(f"Unexpected pipeline output: {result}")


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
    summary_mismatch = False
    for attempt in range(1, num_candidates + 1):
        _log(f"  [Attempt {attempt}/{num_candidates}] Running pipeline")
        result = _run_pipeline(inference_pipeline, row)
        s, message, meta = _extract_pipeline_output(result)
        if summarization is None and s:
            summarization = s
        elif s and summarization and s != summarization:
            summary_mismatch = True
        if isinstance(message, str) and message.strip():
            candidates.append({"text": message.strip(), "meta": meta})
    if summary_mismatch:
        _log("  [Warn] Qwen draft differs across candidates; using the first one.")
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

        meta_for_prompt = candidates[0].get("meta", {}) if candidates else {}
        prompt_text = _build_prompt_text(meta_for_prompt, summarization)
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
