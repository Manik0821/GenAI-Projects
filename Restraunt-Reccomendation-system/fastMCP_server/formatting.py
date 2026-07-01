"""Tool-result formatting helpers for user-facing responses."""

from __future__ import annotations

import json


def extract_tool_result_text(result) -> str:
    """Flatten MCP tool content blocks into a bounded plain-text payload."""
    raw = " ".join(
        item.text if hasattr(item, "text") else str(item)
        for item in result.content
    ) if result.content else "(no result)"
    return raw[:5000]


def format_specialized_result(action: str, tool_output: str) -> str:
    """Render compact user-facing text for specialized tool actions."""
    try:
        data = json.loads(tool_output)
    except Exception:
        return tool_output[:1200]

    if action == "recommend_by_vibe":
        vibe = data.get("vibe_searched", "your vibe")
        vector = data.get("vector_matches", [])[:5]
        structured = data.get("structured_matches", [])[:5]
        lines = [f"Here are quick picks for '{vibe}':"]
        seen = set()

        for item in vector + structured:
            name = item.get("name", "Unknown")
            if name in seen:
                continue
            seen.add(name)
            cuisine = item.get("cuisine") or item.get("food_style") or item.get("type") or "Unknown"
            neighborhood = item.get("neighborhood", "Unknown")
            rating = item.get("rating")
            price = item.get("price_range")
            extra = []
            if rating not in [None, "", "N/A"]:
                extra.append(f"rating {rating}")
            if price not in [None, "", "N/A"]:
                extra.append(str(price))
            suffix = f" — {', '.join(extra)}" if extra else ""
            lines.append(f"- {name} ({neighborhood}), {cuisine}{suffix}")
            if len(seen) >= 5:
                break

        if len(lines) == 1:
            return "I couldn't find strong matches for that vibe. Try a specific vibe like 'moody', 'cozy', or 'zen'."
        return "\n".join(lines)

    if action == "get_restaurant_info":
        if data.get("status") != "found":
            return data.get("message", "I couldn't find that restaurant.")
        result = (data.get("results") or [{}])[0]
        return (
            f"{result.get('name', 'Restaurant')}\n"
            f"- Cuisine: {result.get('cuisine', result.get('food_style', 'Unknown'))}\n"
            f"- Neighborhood: {result.get('neighborhood', result.get('location', 'Unknown'))}\n"
            f"- Rating: {result.get('rating', 'N/A')}\n"
            f"- Price: {result.get('price_range', 'N/A')}"
        )

    if action == "get_review":
        if data.get("status") != "found":
            return data.get("message", "I couldn't find that review.")
        image_info = data.get("image_description") or data.get("image_captions") or ""
        if isinstance(image_info, list):
            image_info = " ".join(str(item) for item in image_info if item)
        response = (
            f"Review for {data.get('restaurant', 'Unknown')}:\n"
            f"- Reviewer: {data.get('reviewer', 'Unknown')}\n"
            f"- Rating: {data.get('rating', 'N/A')}\n"
            f"- Review: {str(data.get('review_text', 'N/A'))[:700]}"
        )
        if image_info and str(image_info).strip().lower() != "n/a":
            response += f"\n- Image insight: {str(image_info)[:500]}"
            return response
        return response

    return tool_output[:1200]
