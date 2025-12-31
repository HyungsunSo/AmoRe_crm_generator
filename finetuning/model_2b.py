def generate_n(model_2b, prompt, n=4):
    if isinstance(prompt, dict):
        persona_id = prompt.get("persona_id")
        product_id = prompt.get("product_id")
        send_purpose = prompt.get("send_purpose", "")
        has_event = bool(prompt.get("has_event"))
        event_content = prompt.get("event_content", "")
    else:
        persona_id = None
        product_id = None
        send_purpose = ""
        has_event = False
        event_content = ""

    persona_summary = f"Persona {persona_id or 'unknown'} prefers a friendly, concise tone."
    product_summary = f"Product {product_id or 'unknown'} should highlight key benefits."

    key_preferences = ["clear benefits", "friendly tone"]
    if send_purpose:
        key_preferences.append(f"purpose: {send_purpose}")

    key_constraints = ["title <= 40 chars", "body <= 350 chars"]
    if has_event and event_content:
        key_constraints.append(f"event: {event_content}")

    analysis = {
        "persona_summary": persona_summary,
        "product_summary": product_summary,
        "key_preferences": key_preferences,
        "key_constraints": key_constraints,
    }

    return {
        "persona_id": persona_id,
        "product_id": product_id,
        "analysis": analysis,
        "n": n,
    }
