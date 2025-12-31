def refine_with_8b(model_8b, generator_input):
    if isinstance(generator_input, dict):
        analysis = generator_input.get("analysis", {})
        n = int(generator_input.get("n", 4) or 4)
    else:
        analysis = {}
        n = 4

    persona_summary = analysis.get("persona_summary", "")
    product_summary = analysis.get("product_summary", "")
    key_preferences = analysis.get("key_preferences", [])

    n = max(1, n)
    candidates = []
    for i in range(n):
        title = f"title: {product_summary[:24]} #{i + 1}"
        body = (
            "body: "
            f"{persona_summary} "
            f"{product_summary} "
            f"Focus on {', '.join(key_preferences) if key_preferences else 'value'}."
        )
        candidates.append({
            "response_id": i,
            "text": f"{title}\n{body}",
        })

    return {"candidates": candidates}
