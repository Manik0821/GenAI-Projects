from app.models import ask_llm
from app.fetchers import fetch_shopify_product_json
from app.sitemap_ingest import discover_products_from_sitemap


def _iter_jsonld_items(json_ld: list) -> list[dict]:
    """Flatten JSON-LD payloads, including @graph blocks, into dict items."""
    items: list[dict] = []
    for item in json_ld or []:
        if isinstance(item, dict):
            items.append(item)
            graph = item.get("@graph")
            if isinstance(graph, list):
                items.extend([g for g in graph if isinstance(g, dict)])
    return items


def _extract_aggregate_rating(record: dict) -> dict | None:
    """Return AggregateRating dict from JSON-LD when present."""
    for item in _iter_jsonld_items(record.get("json_ld", [])):
        item_type = item.get("@type")
        if item_type == "Product" and isinstance(item.get("aggregateRating"), dict):
            return item["aggregateRating"]
        if item_type == "AggregateRating":
            return item
    return None


def _combined_ingredients_text(record: dict) -> str:
    """Return ingredient text from both legacy and current record keys."""
    chunks = []

    # Current ingest key
    sections = record.get("ingredients_sections", [])
    if isinstance(sections, list):
        chunks.extend([str(s) for s in sections if s])

    # Backward-compatible key used in older evaluator logic
    legacy = record.get("ingredients_section", "")
    if legacy:
        chunks.append(str(legacy))

    return "\n".join(chunks).lower()


def _parse_ingredients_list(record: dict) -> list[str]:
    """Parse best-effort ingredient items from extracted ingredient text blocks."""
    text = _combined_ingredients_text(record)
    if not text.strip():
        return []

    normalized = text.replace("ingredients:", " ")
    normalized = normalized.replace("supplement facts:", " ")

    raw_parts = []
    for chunk in normalized.split("\n"):
        for part in chunk.replace(";", ",").split(","):
            cleaned = " ".join(part.split()).strip(" .:-")
            if cleaned:
                raw_parts.append(cleaned)

    noise = {
        "and",
        "or",
        "contains",
        "may contain",
        "serving size",
        "daily value",
    }

    deduped: list[str] = []
    seen = set()
    for item in raw_parts:
        if item in noise:
            continue
        if len(item) < 2:
            continue
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _parse_ingredients_from_text(text: str) -> list[str]:
    """Parse ingredients list from arbitrary text content."""
    if not text:
        return []

    lowered = text.lower()
    snippet = lowered
    marker = "ingredients"
    idx = lowered.find(marker)
    if idx >= 0:
        snippet = lowered[idx: idx + 1500]

    snippet = snippet.replace("ingredients:", " ").replace("supplement facts:", " ")
    raw_parts = []
    for chunk in snippet.split("\n"):
        for part in chunk.replace(";", ",").split(","):
            cleaned = " ".join(part.split()).strip(" .:-")
            if cleaned:
                raw_parts.append(cleaned)

    noise = {
        "and",
        "or",
        "contains",
        "may contain",
        "serving size",
        "daily value",
        "ingredients",
    }

    deduped: list[str] = []
    seen = set()
    for item in raw_parts:
        if item in noise or len(item) < 2:
            continue
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _ingredients_from_shopify_json(url: str) -> list[str]:
    """Extract ingredient-like text from Shopify .js payload body_html."""
    try:
        payload = fetch_shopify_product_json(url)
    except Exception:
        return []
    if not payload:
        return []

    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    body_html = str(data.get("body_html", ""))
    return _parse_ingredients_from_text(body_html)


def _ingredient_similarity_alternatives(record: dict, max_candidates: int = 40, max_results: int = 3) -> list[dict]:
    """Find live Healf alternatives by ingredient overlap from sitemap-discovered products."""
    base_ingredients = _parse_ingredients_list(record)
    if not base_ingredients:
        return []

    base_set = {x.lower() for x in base_ingredients}
    this_url = str(record.get("canonical_url", ""))

    try:
        product_urls = discover_products_from_sitemap(
            "https://healf.com",
            max_child_sitemaps=3,
            max_products=250,
        )
    except Exception:
        return []

    candidate_urls = []
    for url in product_urls:
        if url.rstrip("/") == this_url.rstrip("/"):
            continue
        candidate_urls.append(url)
        if len(candidate_urls) >= max_candidates:
            break

    ranked: list[dict] = []
    for url in candidate_urls:
        cand_ingredients = _ingredients_from_shopify_json(url)
        if not cand_ingredients:
            continue

        cand_set = {x.lower() for x in cand_ingredients}
        overlap = sorted(base_set.intersection(cand_set))
        if not overlap:
            continue

        score = len(overlap)
        ranked.append(
            {
                "name": url.rstrip("/").split("/")[-1].replace("-", " ").title(),
                "url": url,
                "overlap_count": score,
                "shared_ingredients": overlap[:8],
            }
        )

    ranked.sort(key=lambda x: x["overlap_count"], reverse=True)
    return ranked[:max_results]


def _review_presence(record: dict) -> bool:
    """Determine whether review evidence exists in schema or visible page signals."""
    if _extract_aggregate_rating(record):
        return True

    review_signals = record.get("review_signals", {})
    return bool(
        review_signals.get("has_reviews")
        or review_signals.get("has_reviews_keyword")
    )


def _format_shopify_price(value) -> str | None:
    """Convert Shopify cent-based price values into GBP-style text when possible."""
    try:
        cents = int(value)
    except Exception:
        return None
    return f"£{cents / 100:.2f}"


def _evaluate_product_overview(record: dict) -> dict:
    """Return a comprehensive product snapshot for non-specific user queries."""
    text_blocks = record.get("text_blocks", {})
    title = text_blocks.get("title") or "Unknown"
    subheading = text_blocks.get("subheading")
    headings = text_blocks.get("headings", [])
    ingredients = _parse_ingredients_list(record)

    shopify_payload = record.get("shopify_json") or {}
    shopify_data = shopify_payload.get("data", {}) if isinstance(shopify_payload, dict) else {}
    vendor = shopify_data.get("vendor")
    product_type = shopify_data.get("type") or shopify_data.get("product_type")
    tags = shopify_data.get("tags") or []
    variants = shopify_data.get("variants") or []

    price_text = None
    compare_text = None
    variant_count = len(variants)
    if variants:
        first_variant = variants[0] if isinstance(variants[0], dict) else {}
        price_text = _format_shopify_price(first_variant.get("price"))
        compare_text = _format_shopify_price(first_variant.get("compare_at_price"))

    aggregate_rating = _extract_aggregate_rating(record)
    rating_text = None
    review_count_text = None
    if aggregate_rating:
        rating_text = str(aggregate_rating.get("ratingValue", "N/A"))
        review_count_text = str(aggregate_rating.get("reviewCount", "N/A"))

    strengths = []
    weaknesses = []

    if title and title != "Unknown":
        strengths.append("Clear product title available")
    else:
        weaknesses.append("Product title missing")

    if ingredients:
        strengths.append(f"Ingredients section present ({len(ingredients)} entries parsed)")
    else:
        weaknesses.append("Ingredients section not clearly structured")

    if aggregate_rating:
        strengths.append(f"Aggregate review data present ({rating_text}/5 from {review_count_text} reviews)")
    else:
        weaknesses.append("No aggregate review rating found in structured data")

    if subheading:
        strengths.append("Meta description present")
    else:
        weaknesses.append("Meta description missing")

    if headings:
        strengths.append(f"Rich body structure detected ({len(headings)} section headings)")
    else:
        weaknesses.append("Limited visible heading structure")

    lines = ["### Product Overview", ""]
    lines.append(f"- Name: {title}")
    if vendor:
        lines.append(f"- Brand: {vendor}")
    if product_type:
        lines.append(f"- Category: {product_type}")
    if price_text:
        lines.append(f"- Price: {price_text}")
    if compare_text:
        lines.append(f"- Compare-at price: {compare_text}")
    if variant_count:
        lines.append(f"- Variant count: {variant_count}")
    lines.append(f"- URL: {record.get('canonical_url', 'N/A')}")

    lines.append("")
    lines.append("### Aggregate Reviews")
    if aggregate_rating:
        lines.append(f"- Rating: {rating_text}/5")
        lines.append(f"- Review count: {review_count_text}")
    else:
        lines.append("- No aggregate rating/review count found in structured schema")

    lines.append("")
    lines.append("### Strengths")
    lines.extend([f"- {item}" for item in strengths[:6]])

    lines.append("")
    lines.append("### Weaknesses / Gaps")
    if weaknesses:
        lines.extend([f"- {item}" for item in weaknesses[:6]])
    else:
        lines.append("- No major structural gaps detected from extracted data")

    lines.append("")
    lines.append("### Additional Details")
    if ingredients:
        lines.append(f"- Parsed ingredients (first 10): {', '.join(ingredients[:10])}")
    if headings:
        lines.append(f"- Page sections (first 8): {', '.join(headings[:8])}")
    if tags:
        if isinstance(tags, str):
            lines.append(f"- Tags: {tags}")
        elif isinstance(tags, list):
            lines.append(f"- Tags (first 12): {', '.join([str(t) for t in tags[:12]])}")

    sources = ["visible_html", "json_ld"]
    if shopify_data:
        sources.append("shopify_product_json")

    evidence = [
        f"title_present={bool(title and title != 'Unknown')}",
        f"aggregate_rating_present={bool(aggregate_rating)}",
        f"ingredients_count={len(ingredients)}",
        f"headings_count={len(headings)}",
        f"meta_description_present={bool(subheading)}",
    ]

    confidence = "high" if (title and (aggregate_rating or ingredients)) else "medium"
    return _build_response(
        answer="\n".join(lines),
        confidence=confidence,
        sources=sources,
        evidence=evidence,
    )


def _build_response(
    answer: str,
    confidence: str,
    sources: list[str],
    evidence: list[str],
    needs_clarification: bool = False,
    clarification_prompt: str | None = None,
) -> dict:
    """Create a structured response payload for conversational + JSON interfaces."""
    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "evidence": evidence,
        "needs_clarification": needs_clarification,
        "clarification_prompt": clarification_prompt,
    }


def _render_human_response(payload: dict) -> str:
    """Render structured payload into a readable text answer with citations."""
    lines = [payload.get("answer", "")]

    confidence = payload.get("confidence")
    if confidence:
        lines.append(f"\nConfidence: {confidence}")

    sources = payload.get("sources", [])
    if sources:
        lines.append("Sources: " + ", ".join(sources))

    evidence = payload.get("evidence", [])
    if evidence:
        lines.append("Evidence:")
        lines.extend([f"- {item}" for item in evidence[:5]])

    if payload.get("needs_clarification") and payload.get("clarification_prompt"):
        lines.append("\nClarification needed: " + str(payload.get("clarification_prompt")))

    return "\n".join(lines).strip()


def _evaluate_reviews(record: dict) -> dict:
    """Evaluate review presence and return structured output."""
    review_signals = record.get("review_signals", {})
    has_review_widgets = bool(
        review_signals.get("has_reviews", False)
        or review_signals.get("has_reviews_keyword", False)
    )
    detected_widgets = review_signals.get("detected_widgets", [])
    aggregate_rating = _extract_aggregate_rating(record)

    if aggregate_rating:
        rating_value = aggregate_rating.get("ratingValue", "N/A")
        review_count = aggregate_rating.get("reviewCount", "N/A")
        answer = (
            f"✅ Reviews Found! This product has an aggregate rating of {rating_value}/5 "
            f"based on {review_count} structural reviews found in the page schema metadata."
        )
        evidence = [
            f"JSON-LD AggregateRating.ratingValue={rating_value}",
            f"JSON-LD AggregateRating.reviewCount={review_count}",
        ]
        return _build_response(
            answer=answer,
            confidence="high",
            sources=["json_ld"],
            evidence=evidence,
        )

    if has_review_widgets:
        widgets_list = ", ".join([w.strip(".") for w in detected_widgets]) if detected_widgets else "review keywords"
        answer = (
            f"⚠️ Review widgets were detected on the page template ({widgets_list}), "
            f"but no explicit review text counts are visible in the structural schema markup. "
            f"The store likely loads live reviews dynamically using JavaScript apps."
        )
        evidence = [
            f"review_signals.has_reviews_keyword={review_signals.get('has_reviews_keyword', False)}",
            f"review_signals.detected_widgets={detected_widgets}",
        ]
        return _build_response(
            answer=answer,
            confidence="medium",
            sources=["visible_html"],
            evidence=evidence,
        )

    return _build_response(
        answer="❌ No visible user reviews or functional third-party review widgets were detected on this product page layout.",
        confidence="medium",
        sources=["visible_html", "json_ld"],
        evidence=["No AggregateRating found in JSON-LD", "No review keywords/widgets detected in page text signals"],
    )


def _evaluate_ingredient_lookup(record: dict, ingredient_query: str | None) -> dict:
    """Evaluate ingredient presence and return structured output."""
    if not ingredient_query:
        return _build_response(
            answer="❓ I identified an ingredient intent, but I could not isolate the exact ingredient name from your request.",
            confidence="low",
            sources=["router"],
            evidence=["Ingredient intent matched but no entity extracted"],
            needs_clarification=True,
            clarification_prompt="Which exact ingredient should I check? Example: Vitamin D, Magnesium, or Ashwagandha.",
        )

    target = ingredient_query.lower().strip()
    ingredients_text = _combined_ingredients_text(record)
    page_text = record.get("text_blocks", {}).get("page_text", "").lower()

    if target in ingredients_text:
        return _build_response(
            answer=f"🎯 Match Found! '{ingredient_query}' is explicitly listed within the product's dedicated ingredient context segment.",
            confidence="high",
            sources=["ingredients_section"],
            evidence=[f"Matched '{target}' in extracted ingredients sections"],
        )

    if target in page_text:
        return _build_response(
            answer=(
                f"🔍 Partial Match: '{ingredient_query}' was not identified in a formal ingredients table, "
                f"but the term appears elsewhere within the visible description text blocks on this page."
            ),
            confidence="medium",
            sources=["visible_html"],
            evidence=[f"Matched '{target}' in page_text but not in ingredients section"],
        )

    return _build_response(
        answer=f"❌ Discovered no mention of '{ingredient_query}' within the extracted text data or component blocks for this specific product layout.",
        confidence="medium",
        sources=["ingredients_section", "visible_html"],
        evidence=[f"No match for '{target}' in ingredients sections", f"No match for '{target}' in page text"],
    )


def _evaluate_page_improvement_audit(record: dict, user_question: str = "") -> dict:
    """Evaluate page quality and return structured output with deterministic + LLM reasoning."""
    text_blocks = record.get("text_blocks", {})
    json_ld = record.get("json_ld", [])

    title = text_blocks.get("title")
    subheading = text_blocks.get("subheading")
    headings = text_blocks.get("headings", [])
    ingredients = _combined_ingredients_text(record)

    issues = []
    successes = []

    if not title:
        issues.append("Missing standard H1 product page heading tag.")
    else:
        successes.append(f"Valid Product H1 Identified: '{title}'")

    if not subheading:
        issues.append("Missing descriptive HTML meta-description tag for search engines.")

    if len(headings) < 2:
        issues.append("Low contextual content structure (Found fewer than 2 sub-headings (H2/H3)).")

    if not ingredients:
        issues.append("No isolated product ingredients, metrics, or material tabs discovered.")

    has_product_schema = any(
        isinstance(item, dict) and item.get("@type") == "Product"
        for item in _iter_jsonld_items(json_ld)
    )
    if not has_product_schema:
        issues.append("Missing JSON-LD structured Product metadata schemas needed for Google Rich Snippets.")
    else:
        successes.append("Valid JSON-LD structured product schema object is integrated.")

    if not _review_presence(record):
        issues.append("No active app integration hooks discovered for user social proof or rating displays.")

    report = ["### 📊 Page Improvement Audit Report\n"]
    if successes:
        report.append("**Strengths Identified:**")
        report.extend([f"* {s}" for s in successes])
        report.append("")

    if issues:
        report.append("**Critical Optimization Gaps Found:**")
        report.extend([f"⚠️ {i}" for i in issues])
    else:
        report.append("🎉 Outstanding! This product page layout fulfills all structural content checks successfully.")

    llm_used = False
    try:
        llm_prompt = (
            "You are auditing a Healf product page. Provide concise, practical recommendations.\n"
            "Use only the supplied facts.\n\n"
            f"User question: {user_question or 'What can I improve on this page?'}\n"
            f"Title: {title or 'missing'}\n"
            f"Meta description present: {'yes' if subheading else 'no'}\n"
            f"Number of H2/H3 headings: {len(headings)}\n"
            f"Ingredients section present: {'yes' if bool(ingredients.strip()) else 'no'}\n"
            f"Product schema present: {'yes' if has_product_schema else 'no'}\n"
            f"Review evidence present: {'yes' if _review_presence(record) else 'no'}\n"
            f"Detected issues: {issues}\n"
            f"Detected strengths: {successes}\n\n"
            "Return 4-6 bullet points with priority order."
        )
        llm_summary = ask_llm(llm_prompt).strip()
        if llm_summary:
            llm_used = True
            report.append("")
            report.append("**LLM Reasoning Summary:**")
            report.append(llm_summary)
    except Exception as exc:
        report.append("")
        report.append(f"Note: LLM summary unavailable ({exc}). Deterministic audit is still provided above.")

    evidence = [
        f"title_present={bool(title)}",
        f"meta_description_present={bool(subheading)}",
        f"heading_count={len(headings)}",
        f"ingredients_present={bool(ingredients.strip())}",
        f"product_schema_present={has_product_schema}",
        f"review_evidence_present={_review_presence(record)}",
    ]

    sources = ["visible_html", "json_ld"]
    if record.get("shopify_json") is not None:
        sources.append("shopify_product_json")
    if llm_used:
        sources.append("nvidia_llm")

    confidence = "high" if len(issues) == 0 or llm_used else "medium"
    return _build_response(
        answer="\n".join(report),
        confidence=confidence,
        sources=sources,
        evidence=evidence,
    )


def _evaluate_ingredients_list(record: dict, include_alternatives: bool = False) -> dict:
    """List all parsed ingredients and optionally suggest alternatives."""
    ingredients = _parse_ingredients_list(record)
    if not ingredients:
        return _build_response(
            answer="I could not confidently extract a full ingredients list from this product page.",
            confidence="low",
            sources=["ingredients_section", "visible_html"],
            evidence=["No parseable ingredient entries found in extracted ingredient text blocks"],
            needs_clarification=True,
            clarification_prompt="Try asking a specific ingredient check, e.g., 'Does this product have Vitamin D?'",
        )

    alternatives = _ingredient_similarity_alternatives(record) if include_alternatives else []

    lines = ["### Ingredients detected", ""]
    lines.extend([f"- {item}" for item in ingredients[:40]])

    evidence = [f"ingredients_count={len(ingredients)}"]
    sources = ["ingredients_section", "visible_html"]

    if include_alternatives and alternatives:
        lines.append("")
        lines.append("### Alternative Healf products with similar ingredients")
        lines.append("")
        for alt in alternatives:
            shared = ", ".join(alt.get("shared_ingredients", [])[:5])
            lines.append(
                f"- {alt['name']} ({alt['url']}) - shared ingredients: {shared}"
            )
        evidence.append(f"alternatives_found={len(alternatives)}")
        sources.extend(["sitemap", "live_product_pages"])
        confidence = "medium"
    elif include_alternatives:
        lines.append("")
        lines.append("No strong alternatives were found in the sampled live Healf product set.")
        evidence.append("alternatives_found=0")
        confidence = "medium"
    else:
        confidence = "high"

    return _build_response(
        answer="\n".join(lines),
        confidence=confidence,
        sources=sources,
        evidence=evidence,
    )

def handle_reviews_check(record: dict) -> str:
    """
    Evaluates review signals extracted from the HTML and metadata.
    """
    return _evaluate_reviews(record)["answer"]


def handle_ingredient_lookup(record: dict, ingredient_query: str) -> str:
    """
    Scans the extracted page text and specialized sections to verify ingredient presence.
    """
    return _evaluate_ingredient_lookup(record, ingredient_query)["answer"]


def handle_page_improvement_audit(record: dict, user_question: str = "") -> str:
    """
    Audits the structured data payload to find common conversion or optimization gaps.
    """
    return _evaluate_page_improvement_audit(record, user_question=user_question)["answer"]


def execute_agent_flow(record: dict, question: str) -> str:
    """
    Orchestrates the entire flow: routes the text, hooks into the right evaluator,
    and returns the structured natural language response.
    """
    # 1. Route the question using your Router logic
    from app.router import route_question  # Assuming your router file is named app/router.py
    
    route = route_question(question)
    
    payload = execute_agent_flow_structured(record, question)
    return _render_human_response(payload)


def execute_agent_flow_structured(record: dict, question: str) -> dict:
    """Structured version of the agent flow for API/JSON/chat interfaces."""
    from app.router import route_question

    route = route_question(question)

    if route.intent == "reviews_check":
        payload = _evaluate_reviews(record)
    elif route.intent == "product_overview":
        payload = _evaluate_product_overview(record)
    elif route.intent == "ingredients_list":
        payload = _evaluate_ingredients_list(record, include_alternatives=False)
    elif route.intent == "ingredients_with_alternatives":
        payload = _evaluate_ingredients_list(record, include_alternatives=True)
    elif route.intent == "ingredient_lookup":
        payload = _evaluate_ingredient_lookup(record, route.ingredient_query)
    elif route.intent == "page_improvement_audit":
        payload = _evaluate_page_improvement_audit(record, user_question=question)
    else:
        payload = _build_response(
            answer="Unknown agent destination routing error encountered.",
            confidence="low",
            sources=["router"],
            evidence=[f"Unsupported route intent: {route.intent}"],
        )

    payload["intent"] = route.intent
    payload["ingredient_query"] = route.ingredient_query
    return payload
