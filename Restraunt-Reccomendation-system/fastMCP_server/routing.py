"""Routing and request-understanding helpers for the restaurant assistant."""

from __future__ import annotations

import re

PREFERENCE_KEYWORDS = {
    "spicy",
    "healthy",
    "light",
    "fresh",
    "protein",
    "vegetarian",
    "vegan",
    "gluten-free",
    "gluten free",
    "low carb",
    "keto",
    "high protein",
    "comfort",
    "savory",
    "sweet",
    "umami",
    "cozy",
    "zen",
    "romantic",
    "moody",
}

CUISINE_KEYWORDS = {
    "french",
    "italian",
    "japanese",
    "sushi",
    "ramen",
    "chinese",
    "szechuan",
    "korean",
    "mexican",
    "oaxacan",
    "indian",
    "persian",
    "mediterranean",
    "vietnamese",
    "thai",
    "american",
    "seafood",
    "lebanese",
    "tapas",
    "bbq",
    "barbecue",
    "fusion",
    "pizza",
    "burger",
    "steak",
    "asian",
    "latin",
    "spanish",
    "greek",
    "turkish",
}


def _extract_vibe_argument(user_message: str) -> str:
    """Extract a concise recommendation query from free-form preference text."""
    lower = user_message.strip().lower()

    location = ""
    location_match = re.search(r"\b(?:in|near)\s+([a-z][a-z\s\-()]{1,40})(?:\s|$)", lower)
    if location_match:
        location = location_match.group(1).strip()

    matched_cuisines = [k for k in CUISINE_KEYWORDS if k in lower]
    if matched_cuisines:
        core = ", ".join(dict.fromkeys(matched_cuisines))
        return f"{core} in {location}" if location else core

    matched_prefs = [k for k in PREFERENCE_KEYWORDS if k in lower]
    if matched_prefs:
        core = ", ".join(dict.fromkeys(matched_prefs))
        return f"{core} in {location}" if location else core

    something_match = re.search(r"something\s+([a-z\-]+)", lower)
    if something_match:
        descriptor = something_match.group(1).strip()
        return f"{descriptor} in {location}" if location else descriptor

    return user_message.strip()


def _heuristic_route(user_message: str) -> tuple[str, str]:
    """Route obvious intents without LLM calls to minimize response latency."""
    text = user_message.strip()
    lower = text.lower()

    if any(k in lower for k in ["review", "reviews"]):
        return "get_review", text
    if any(k in lower for k in ["tell me about", "about", "details", "info", "information"]):
        return "get_restaurant_info", text
    if any(
        k in lower
        for k in [
            "vibe",
            "moody",
            "romantic",
            "zen",
            "cozy",
            "ambience",
            "atmosphere",
            "spot",
            "restaurants",
            "recommend",
            "eat",
            "hungry",
            "craving",
            "something",
            "cuisine",
            "food",
            *PREFERENCE_KEYWORDS,
            *CUISINE_KEYWORDS,
        ]
    ):
        return "recommend_by_vibe", _extract_vibe_argument(text)
    return "fallback_react", text


def _thinking_message_for_action(action: str, query: str = "") -> str:
    """Return a dynamic thinking message based on action and query context."""
    query_lower = query.lower().strip()

    if action == "recommend_by_vibe":
        location_match = re.search(r"\bin\s+([a-z][a-z\s\-()]{1,40})$", query_lower)
        location = location_match.group(1).strip() if location_match else ""

        if any(cuisine in query_lower for cuisine in CUISINE_KEYWORDS):
            for cuisine in CUISINE_KEYWORDS:
                if cuisine in query_lower:
                    if location:
                        return f"[*] Searching for {cuisine.title()} cuisines in {location.title()}..."
                    return f"[*] Searching for restaurants that serve {cuisine.title()} cuisines..."
        if any(pref in query_lower for pref in PREFERENCE_KEYWORDS):
            for pref in PREFERENCE_KEYWORDS:
                if pref in query_lower:
                    if location:
                        return f"[*] Finding {pref} restaurants in {location.title()}..."
                    return f"[*] Finding {pref} restaurants for you..."
        if location:
            return f"[*] Searching for restaurants in {location.title()}..."
        return "[*] Searching for restaurants matching your taste..."
    if action == "get_restaurant_info":
        return f"[*] Looking up details for {query}..."
    if action == "get_review":
        return f"[*] Fetching reviews and dining experience for {query}..."
    return "[*] Processing your request..."
