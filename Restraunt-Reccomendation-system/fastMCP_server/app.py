import json
import os
import re

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "").rstrip("/")

if not BACKEND_API_URL:
    print("WARNING: BACKEND_API_URL is not set.", flush=True)


PREFERENCE_KEYWORDS = {
    "spicy", "healthy", "light", "fresh", "protein", "vegetarian", "vegan",
    "gluten-free", "gluten free", "low carb", "keto", "high protein",
    "comfort", "savory", "sweet", "umami", "cozy", "zen", "romantic", "moody",
}

CUISINE_KEYWORDS = {
    "french", "italian", "japanese", "sushi", "ramen", "chinese", "szechuan",
    "korean", "mexican", "oaxacan", "indian", "persian", "mediterranean",
    "vietnamese", "thai", "american", "seafood", "lebanese", "tapas",
    "bbq", "barbecue", "fusion", "pizza", "burger", "steak", "asian",
    "latin", "spanish", "greek", "turkish",
}


def extract_vibe_argument(user_message: str) -> str:
    lower = str(user_message).strip().lower()

    location = ""
    location_match = re.search(
        r"\b(?:in|near|around|close to)\s+([a-z][a-z\s\-()]{1,40})(?:\s|$)",
        lower,
    )

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

    something_match = re.search(r"\bsomething\s+([a-z\-]+)", lower)
    if something_match:
        descriptor = something_match.group(1).strip()
        return f"{descriptor} in {location}" if location else descriptor

    return str(user_message).strip()


def heuristic_route(user_message: str) -> tuple[str, str]:
    text = str(user_message).strip()
    lower = text.lower()

    if any(k in lower for k in ["review", "reviews"]):
        return "get_review", text

    if any(k in lower for k in ["tell me about", "details", "info", "information"]):
        return "get_restaurant_info", text

    if any(
        k in lower
        for k in [
            "about", "vibe", "moody", "romantic", "zen", "cozy",
            "ambience", "atmosphere", "spot", "restaurant", "restaurants",
            "recommend", "eat", "hungry", "craving", "something",
            "cuisine", "food", *PREFERENCE_KEYWORDS, *CUISINE_KEYWORDS,
        ]
    ):
        return "recommend_by_vibe", extract_vibe_argument(text)

    return "recommend_by_vibe", text


def format_result(action: str, tool_output: str) -> str:
    try:
        data = json.loads(tool_output)
    except Exception:
        return str(tool_output)[:1500]

    if action == "recommend_by_vibe":
        vibe = data.get("vibe_searched", "your request")
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
            neighborhood = item.get("neighborhood") or item.get("location") or "Unknown"
            rating = item.get("rating", "N/A")
            price = item.get("price_range", "N/A")

            lines.append(
                f"- {name} ({neighborhood}) — {cuisine}, Rating: {rating}, Price: {price}"
            )

            if len(seen) >= 5:
                break

        if len(lines) == 1:
            return "I couldn't find strong matches. Try 'moody restaurants', 'french in pasadena', or 'fresh seafood'."

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
            f"- Price: {result.get('price_range', 'N/A')}\n"
            f"- Signature dish: {result.get('signature_dish', 'N/A')}"
        )

    if action == "get_review":
        if data.get("status") != "found":
            return data.get("message", "I couldn't find that review.")

        image_info = data.get("image_description") or data.get("image_captions") or ""
        if isinstance(image_info, list):
            image_info = " ".join(str(x) for x in image_info if x)

        response = (
            f"Review for {data.get('restaurant', 'Unknown')}:\n"
            f"- Reviewer: {data.get('reviewer', 'Unknown')}\n"
            f"- Rating: {data.get('rating', 'N/A')}\n"
            f"- Review: {str(data.get('review_text', 'N/A'))[:700]}"
        )

        if image_info and str(image_info).lower() != "n/a":
            response += f"\n- Image insight: {str(image_info)[:500]}"

        return response

    return str(tool_output)[:1500]


async def call_backend(action: str, argument: str) -> str:
    if not BACKEND_API_URL:
        return "BACKEND_API_URL is not configured."

    endpoint_map = {
        "recommend_by_vibe": "/recommend_by_vibe",
        "get_restaurant_info": "/get_restaurant_info",
        "get_review": "/get_review",
    }

    endpoint = endpoint_map.get(action, "/recommend_by_vibe")

    payload = (
        {"vibe": argument}
        if action == "recommend_by_vibe"
        else {"restaurant_name": argument}
    )

    print("Calling backend:", f"{BACKEND_API_URL}{endpoint}", payload, flush=True)

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{BACKEND_API_URL}{endpoint}", json=payload)
        response.raise_for_status()
        data = response.json()

    return data.get("result", "")


async def chat_with_agent(user_message: str, history: list) -> tuple[str, str]:
    action, argument = heuristic_route(user_message)
    output = await call_backend(action, argument)
    return format_result(action, output), action


async def handle_chat(user_message, history):
    print("handle_chat called:", repr(user_message), flush=True)

    if history is None:
        history = []

    if not user_message or not str(user_message).strip():
        return history

    user_message = str(user_message).strip()
    history = list(history)

    history.append({"role": "user", "content": user_message})

    try:
        response_text, _ = await chat_with_agent(user_message, history)
    except Exception as exc:
        print(f"Chat error: {repr(exc)}", flush=True)
        response_text = (
            "I couldn't fetch restaurant data right now. "
            "Please check the backend service."
        )

    history.append({"role": "assistant", "content": response_text})
    return history


async def handle_moody(history):
    return await handle_chat("Find me some moody restaurants", history)


async def handle_iron(history):
    return await handle_chat("Tell me about Iron & Embers", history)


async def handle_zen(history):
    return await handle_chat("What's a zen dining experience in Little Tokyo?", history)


with gr.Blocks(title="Connoisseur Companion") as demo:
    gr.Markdown(
        "# Connoisseur Companion\n"
        "Your AI guide to California's restaurant scene. Ask me about restaurants by name, cuisine, or vibe!"
    )

    chatbot = gr.Chatbot(height=500, type="messages", allow_tags=False)

    msg_input = gr.Textbox(
        label="Ask about restaurants",
        placeholder='e.g., "Find me a moody spot in DTLA" or "Tell me about Sakura Garden"',
    )

    with gr.Row():
        btn1 = gr.Button("Find moody restaurants", size="sm")
        btn2 = gr.Button("Tell me about Iron & Embers", size="sm")
        btn3 = gr.Button("Zen dining in Little Tokyo?", size="sm")

    msg_input.submit(
        handle_chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot],
        queue=True,
    )

    btn1.click(handle_moody, inputs=[chatbot], outputs=[chatbot], queue=True)
    btn2.click(handle_iron, inputs=[chatbot], outputs=[chatbot], queue=True)
    btn3.click(handle_zen, inputs=[chatbot], outputs=[chatbot], queue=True)


if __name__ == "__main__":
    print("Starting Connoisseur Companion...", flush=True)
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        share=False,
        server_name="0.0.0.0",
    )