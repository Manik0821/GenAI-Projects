import argparse
import json
import sys

from app.evaluator import execute_agent_flow, execute_agent_flow_structured
from app.ingest import ingest_product_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Healf Product Intelligence Agent (MVP)",
    )
    parser.add_argument(
        "--url",
        required=False,
        help="Healf product URL",
    )
    parser.add_argument(
        "--question",
        required=False,
        help="Natural language question about the product page",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn chat mode for follow-up questions",
    )
    return parser.parse_args()


def _print_chat_help() -> None:
    print("Commands:")
    print("  /url <healf-pdp-url>  Switch to a new product URL")
    print("  /quit                 Exit chat")


def run_chat(url: str) -> int:
    try:
        record = ingest_product_url(url)
    except Exception as exc:
        print(f"Failed to ingest URL: {exc}", file=sys.stderr)
        return 1

    print("Healf Product Intelligence Agent (chat mode)")
    print(f"Active URL: {record.get('canonical_url', url)}")
    _print_chat_help()

    pending_clarification = None
    history: list[dict] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            return 0

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye.")
            return 0

        if user_input.lower().startswith("/url "):
            new_url = user_input[5:].strip()
            if not new_url:
                print("Please provide a product URL after /url")
                continue
            try:
                record = ingest_product_url(new_url)
                pending_clarification = None
                history.clear()
                print(f"Switched URL: {record.get('canonical_url', new_url)}")
            except Exception as exc:
                print(f"Failed to ingest URL: {exc}")
            continue

        question = user_input
        if pending_clarification:
            question = f"Does this product have {user_input} in it?"

        payload = execute_agent_flow_structured(record, question)
        history.append({"question": question, "intent": payload.get("intent")})

        print("\nAgent:")
        print(payload.get("answer", ""))
        print(f"\nConfidence: {payload.get('confidence', 'unknown')}")

        sources = payload.get("sources", [])
        if sources:
            print("Sources: " + ", ".join(sources))

        evidence = payload.get("evidence", [])
        if evidence:
            print("Evidence:")
            for item in evidence[:4]:
                print(f"- {item}")

        if payload.get("needs_clarification"):
            pending_clarification = payload.get("intent")
            print("\nClarification:")
            print(payload.get("clarification_prompt", "Please clarify your request."))
        else:
            pending_clarification = None


def main() -> int:
    args = parse_args()

    if args.chat:
        if not args.url:
            print("--chat requires --url", file=sys.stderr)
            return 2
        return run_chat(args.url)

    if not args.url or not args.question:
        print("Non-chat mode requires both --url and --question", file=sys.stderr)
        return 2

    try:
        record = ingest_product_url(args.url)
    except Exception as exc:
        print(f"Agent failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        try:
            payload = execute_agent_flow_structured(record, args.question)
        except Exception as exc:
            print(f"Agent failed: {exc}", file=sys.stderr)
            return 1
        payload = {
            "url": record.get("canonical_url", args.url),
            "question": args.question,
            "answer": payload.get("answer", ""),
            "intent": payload.get("intent"),
            "confidence": payload.get("confidence"),
            "sources": payload.get("sources", []),
            "evidence": payload.get("evidence", []),
            "needs_clarification": payload.get("needs_clarification", False),
            "clarification_prompt": payload.get("clarification_prompt"),
            "signals": {
                "title": record.get("text_blocks", {}).get("title"),
                "has_shopify_json": record.get("shopify_json") is not None,
                "ingredients_sections": len(record.get("ingredients_sections", [])),
                "has_review_keyword": record.get("review_signals", {}).get("has_reviews_keyword", False),
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    try:
        answer = execute_agent_flow(record, args.question)
    except Exception as exc:
        print(f"Agent failed: {exc}", file=sys.stderr)
        return 1

    print("Question:")
    print(args.question)
    print("\nAnswer:")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
