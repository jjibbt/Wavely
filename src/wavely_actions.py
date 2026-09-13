"""Gesture names and Home Assistant webhook payloads."""

GESTURES = (
    "OPEN_PALM", "FIST", "PEACE", "POINT", "THUMBS_UP", "THUMBS_DOWN",
    "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN",
    "BOTH_OPEN_PALMS", "BOTH_FISTS", "BOTH_THUMBS_UP", "BOTH_THUMBS_DOWN",
    "LEFT_FIST_RIGHT_OPEN", "LEFT_OPEN_RIGHT_FIST", "HANDS_UP", "HANDS_DOWN",
    "SPREAD_HANDS", "CLOSE_HANDS",
)


def payload_for_gesture(person, hand, gesture, actions):
    payload = {"person": person, "hand": hand, "gesture": gesture}
    mapped = next((item for item in actions if isinstance(item, dict) and item.get("gesture") == gesture), None)
    if mapped and mapped.get("id"):
        payload["action"] = mapped["id"]
        if mapped.get("entity_id"):
            payload["entity_id"] = mapped["entity_id"]
    return payload
