import csv
import json
import os
import re
import urllib.error
import urllib.request

from model_2b import generate_n
from model_8b import refine_with_8b

MODEL_BASE_DIR = "../models"


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedupe(items):
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


def _candidate_text(candidate):
    if isinstance(candidate, dict):
        return candidate.get("text", "") or candidate.get("response", "")
    return candidate


def _format_analysis(analysis):
    if isinstance(analysis, dict):
        lines = ["ANALYSIS"]
        persona_summary = analysis.get("persona_summary", "")
        product_summary = analysis.get("product_summary", "")
        key_preferences = analysis.get("key_preferences", [])
        key_constraints = analysis.get("key_constraints", [])

        if persona_summary:
            lines.append(f"persona_summary: {persona_summary}")
        if product_summary:
            lines.append(f"product_summary: {product_summary}")
        if key_preferences:
            lines.append("key_preferences:")
            for item in key_preferences:
                lines.append(f"- {item}")
        if key_constraints:
            lines.append("key_constraints:")
            for item in key_constraints:
                lines.append(f"- {item}")

        return "\n".join(lines)
    return str(analysis)


def _normalize_candidates(candidates):
    if isinstance(candidates, dict) and "candidates" in candidates:
        candidates = candidates["candidates"]

    if candidates is None:
        return []

    normalized = []
    for idx, item in enumerate(_ensure_list(candidates)):
        if isinstance(item, dict):
            text = _candidate_text(item)
            response_id = item.get("response_id", idx)
        else:
            text = str(item)
            response_id = idx

        if not text:
            continue
        normalized.append({"response_id": response_id, "text": text})
    return normalized


def _resolve_choice_index(choice, candidates):
    for idx, candidate in enumerate(candidates):
        if candidate.get("response_id", idx) == choice:
            return idx
    if 0 <= choice < len(candidates):
        return choice
    raise ValueError(f"Evaluator index out of range: {choice}")


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


def _call_gpt4o(analysis, candidates):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    print(f"candidates : {candidates}")
    candidate_lines = []
    for i, candidate in enumerate(candidates):
        response_id = candidate.get("response_id", i)
        candidate_lines.append(f"[{response_id}] {candidate.get('text', '')}")
    candidate_block = "\n\n".join(candidate_lines)

    system_prompt = (
        "당신은 DPO 학습용 최종 답변을 고르는 평가자입니다. "
        "명확성, 요청된 출력 형식(제목/본문 및 길이 제한) 준수, 톤 일치, "
        "전반적인 유용성을 기준으로 가장 좋은 후보 1개를 선택하세요."
    )
    user_prompt = (
        "분석:\n"
        f"{_format_analysis(analysis)}\n\n"
        "후보:\n"
        f"{candidate_block}\n\n"
        "가장 좋은 후보의 response_id만 정수로 반환하세요. "
        "동점이면 가장 낮은 response_id를 선택하세요."
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
    return _resolve_choice_index(choice, candidates)


def gpt4o_pick_best(analysis, candidates):
    if not candidates:
        return ""

    best_index = _call_gpt4o(analysis, candidates)
    return candidates[best_index]


def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "chosen", "rejected"])
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append({
                "prompt": row.get("prompt", ""),
                "chosen": row.get("chosen", ""),
                "rejected": row.get("rejected", ""),
            })
        return rows


def _next_cycle_id(data_dir):
    if not os.path.isdir(data_dir):
        return 1

    cycle_ids = []
    for name in os.listdir(data_dir):
        match = re.match(r"cycle_(\d+)\.csv$", name)
        if match:
            cycle_ids.append(int(match.group(1)))
    return max(cycle_ids) + 1 if cycle_ids else 1


def generate_cycle(cycle_id, model_2b, model_8b, prompts):
    out_path = f"finetuning_data/cycle_{cycle_id:02d}.csv"
    rows = _load_existing_rows(out_path)
    print(f"[Cycle {cycle_id:02d}] Loaded existing rows: {len(rows)}")
    for p in prompts:
        print("[Stage 1] Analyzer: start")
        analysis_result = generate_n(model_2b, p, n=4)
        if isinstance(analysis_result, dict) and "analysis" in analysis_result:
            analysis = analysis_result.get("analysis")
            n_candidates = analysis_result.get("n", 4)
        else:
            analysis = analysis_result
            n_candidates = 4

        if not analysis:
            print("[Stage 1] Analyzer: empty analysis, skipping")
            continue

        generator_input = {"analysis": analysis, "n": n_candidates}
        print(f"[Stage 1] Analyzer: done (n={n_candidates})")
        print("[Stage 2] Generator: start")
        raw_candidates = refine_with_8b(model_8b, generator_input)
        candidates = _dedupe(_normalize_candidates(raw_candidates))
        if not candidates:
            print("[Stage 2] Generator: no candidates, skipping")
            continue
        print(f"[Stage 2] Generator: done (candidates={len(candidates)})")

        print("[Stage 3] Discriminator: start")
        best = gpt4o_pick_best(analysis, candidates)
        if not best:
            print("[Stage 3] Discriminator: no best candidate, skipping")
            continue
        print("[Stage 3] Discriminator: done")

        analysis_prompt = _format_analysis(analysis)
        best_text = _candidate_text(best)

        for r in candidates:
            if r != best:
                rejected_text = _candidate_text(r)
                if not rejected_text:
                    continue
                rows.append({
                    "prompt": analysis_prompt,
                    "chosen": best_text,
                    "rejected": rejected_text
                })

    save_csv(out_path, rows)
    print(f"[Cycle {cycle_id:02d}] Saved rows: {len(rows)} -> {out_path}")


if __name__ == "__main__":
    raise RuntimeError(
        "Call generate_cycle(cycle_id, model_2b, model_8b, prompts) from your runner."
    )
