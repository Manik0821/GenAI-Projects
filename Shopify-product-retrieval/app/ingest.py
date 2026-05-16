from app.fetchers import fetch_html, fetch_shopify_product_json, make_soup
from app.parsers import extract_json_ld, extract_text_blocks, extract_ingredients_section, detect_review_signals
from app.url_utils import normalize_product_url, extract_locale_and_handle


def ingest_product_url(url: str) -> dict:
    clean_url = normalize_product_url(url)
    locale, handle = extract_locale_and_handle(clean_url)

    html_result = fetch_html(clean_url)
    soup = make_soup(html_result["html"])

    text_blocks = extract_text_blocks(soup)
    sections = text_blocks.get("sections", {})
    page_text = text_blocks.get("page_text", "")

    return {
        "input_url": clean_url,
        "canonical_url": clean_url,
        "locale": locale,
        "handle": handle,
        "variant_id": None,
        "shopify_json": fetch_shopify_product_json(clean_url),
        "json_ld": extract_json_ld(soup),
        "text_blocks": text_blocks,
        "ingredients_sections": extract_ingredients_section(page_text, sections),
        "review_signals": detect_review_signals(page_text),
    }