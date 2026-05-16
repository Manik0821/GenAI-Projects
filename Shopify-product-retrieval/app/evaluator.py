import json

def handle_reviews_check(record: dict) -> str:
    """
    Evaluates review signals extracted from the HTML and metadata.
    """
    # Check for raw structural tags extracted by detect_review_signals
    review_signals = record.get("review_signals", {})
    has_review_widgets = review_signals.get("has_reviews", False)
    detected_widgets = review_signals.get("detected_widgets", [])
    
    # Also inspect JSON-LD blocks for Schema.org AggregateRating data
    json_ld_items = record.get("json_ld", [])
    aggregate_rating = None
    
    for item in json_ld_items:
        # Some schemas are nested or inside graphs
        if isinstance(item, dict):
            if item.get("@type") == "Product" and "aggregateRating" in item:
                aggregate_rating = item["aggregateRating"]
                break
            elif item.get("@type") == "AggregateRating":
                aggregate_rating = item
                break

    # Build the analytical response
    if aggregate_rating:
        rating_value = aggregate_rating.get("ratingValue", "N/A")
        review_count = aggregate_rating.get("reviewCount", "N/A")
        return (
            f"✅ Reviews Found! This product has an aggregate rating of {rating_value}/5 "
            f"based on {review_count} structural reviews found in the page schema metadata."
        )
        
    if has_review_widgets:
        widgets_list = ", ".join([w.strip(".") for w in detected_widgets])
        return (
            f"⚠️ Review widgets were detected on the page template ({widgets_list}), "
            f"but no explicit review text counts are visible in the structural schema markup. "
            f"The store likely loads live reviews dynamically using JavaScript apps."
        )
        
    return "❌ No visible user reviews or functional third-party review widgets were detected on this product page layout."


def handle_ingredient_lookup(record: dict, ingredient_query: str) -> str:
    """
    Scans the extracted page text and specialized sections to verify ingredient presence.
    """
    if not ingredient_query:
        return "❓ I identified an ingredient intent, but I could not isolate the exact ingredient name from your request."
    
    target = ingredient_query.lower().strip()
    
    # Search spaces
    ingredients_text = record.get("ingredients_section", "").lower()
    page_text = record.get("text_blocks", {}).get("page_text", "").lower()
    
    # Exact or substring block match evaluation
    if target in ingredients_text:
        return f"🎯 Match Found! '{ingredient_query}' is explicitly listed within the product's dedicated ingredient context segment."
        
    if target in page_text:
        return (
            f"🔍 Partial Match: '{ingredient_query}' was not identified in a formal ingredients table, "
            f"but the term appears elsewhere within the visible description text blocks on this page."
        )
        
    return f"❌ Discovered no mention of '{ingredient_query}' within the extracted text data or component blocks for this specific product layout."


def handle_page_improvement_audit(record: dict) -> str:
    """
    Audits the structured data payload to find common conversion or optimization gaps.
    """
    text_blocks = record.get("text_blocks", {})
    json_ld = record.get("json_ld", [])
    review_signals = record.get("review_signals", {})
    
    title = text_blocks.get("title")
    subheading = text_blocks.get("subheading")
    headings = text_blocks.get("headings", [])
    ingredients = record.get("ingredients_section", "")
    
    issues = []
    successes = []
    
    # 1. Audit Title & Metadata
    if not title:
        issues.append("Missing standard H1 product page heading tag.")
    else:
        successes.append(f"Valid Product H1 Identified: '{title}'")
        
    if not subheading:
        issues.append("Missing descriptive HTML meta-description tag for search engines.")
        
    # 2. Audit Structured Content Density
    if len(headings) < 2:
        issues.append("Low contextual content structure (Found fewer than 2 sub-headings (H2/H3)).")
        
    if not ingredients:
        issues.append("No isolated product ingredients, metrics, or material tabs discovered.")
        
    # 3. Audit Rich Schema Data
    has_product_schema = any(isinstance(item, dict) and item.get("@type") == "Product" for item in json_ld)
    if not has_product_schema:
        issues.append("Missing JSON-LD structured Product metadata schemas needed for Google Rich Snippets.")
    else:
        successes.append("Valid JSON-LD structured product schema object is integrated.")
        
    # 4. Audit Social Proof
    if not review_signals.get("has_reviews", False):
        issues.append("No active app integration hooks discovered for user social proof or rating displays.")

    # Format the Audit Report output
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
        
    return "\n".join(report)


def execute_agent_flow(record: dict, question: str) -> str:
    """
    Orchestrates the entire flow: routes the text, hooks into the right evaluator,
    and returns the structured natural language response.
    """
    # 1. Route the question using your Router logic
    from app.router import route_question  # Assuming your router file is named app/router.py
    
    route = route_question(question)
    
    # 2. Dispatch to the matching executor layer based on determined intent
    if route.intent == "reviews_check":
        return handle_reviews_check(record)
        
    elif route.intent == "ingredient_lookup":
        return handle_ingredient_lookup(record, route.ingredient_query)
        
    elif route.intent == "page_improvement_audit":
        return handle_page_improvement_audit(record)
        
    return "Unknown agent destination routing error encountered."
