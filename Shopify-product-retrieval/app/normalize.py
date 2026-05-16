import re
from bs4 import BeautifulSoup

from app.schemas import ProductRecord, PriceInfo, ReviewInfo, ImageInfo


def normalize_ingestion_result(data: dict) -> ProductRecord:
    text_blocks = data.get("text_blocks", {})
    shopify_json = data.get("shopify_json")
    json_ld = data.get("json_ld", [])
    review_signals = data.get("review_signals", {})
    ingredients_sections = data.get("ingredients_sections", [])

    title = _pick_title(text_blocks, shopify_json, json_ld)
    brand = _pick_brand(shopify_json, json_ld)
    description_text = _pick_description(text_blocks, shopify_json)
    headings = text_blocks.get("headings", [])

    price = _extract_price(shopify_json, json_ld, text_blocks)
    reviews = _extract_reviews(review_signals, json_ld, text_blocks)
    images = _extract_images(shopify_json)

    return ProductRecord(
        source_url=data["input_url"],
        canonical_url=data.get("canonical_url"),
        handle=data["handle"],
        locale=data.get("locale"),
        variant_id=data.get("variant_id"),
        title=title,
        brand=brand,
        description_text=description_text,
        headings=headings,
        ingredients_text=ingredients_sections,
        faq_text=[],
        price=price,
        reviews=reviews,
        images=images,
        shopify_json_present=shopify_json is not None,
        json_ld_present=len(json_ld) > 0,
        raw_shopify_json=shopify_json["data"] if shopify_json else None,
        raw_json_ld=json_ld,
    )


def _pick_title(text_blocks: dict, shopify_json: dict | None, json_ld: list) -> str | None:
    if text_blocks.get("title"):
        return text_blocks["title"]

    if shopify_json and shopify_json.get("data", {}).get("title"):
        return shopify_json["data"]["title"]

    for item in json_ld:
        if isinstance(item, dict) and item.get("name"):
            return item["name"]

    return None


def _pick_brand(shopify_json: dict | None, json_ld: list) -> str | None:
    if shopify_json:
        vendor = shopify_json.get("data", {}).get("vendor")
        if vendor:
            return vendor

    for item in json_ld:
        if isinstance(item, dict):
            brand = item.get("brand")
            if isinstance(brand, dict) and brand.get("name"):
                return brand["name"]
            if isinstance(brand, str):
                return brand

    return None


def _pick_description(text_blocks: dict, shopify_json: dict | None) -> str | None:
    if text_blocks.get("subheading"):
        return text_blocks["subheading"]

    if shopify_json:
        desc = shopify_json.get("data", {}).get("description")
        if desc:
            soup = BeautifulSoup(desc, "html.parser")
            return soup.get_text(" ", strip=True)

    return None


def _extract_price(shopify_json: dict | None, json_ld: list, text_blocks: dict) -> PriceInfo:
    if shopify_json and shopify_json.get("data", {}).get("variants"):
        variants = shopify_json["data"]["variants"]
        prices = []
        for v in variants:
            price = v.get("price")
            if isinstance(price, int):
                prices.append(price / 100.0)
            elif isinstance(price, float):
                prices.append(price)

        if prices:
            return PriceInfo(
                min_price=min(prices),
                max_price=max(prices),
                raw_price_text=f"{min(prices)} - {max(prices)}" if len(set(prices)) > 1 else str(prices[0])
            )

    for item in json_ld:
        if isinstance(item, dict):
            offers = item.get("offers")
            if isinstance(offers, dict):
                price = offers.get("price")
                currency = offers.get("priceCurrency")
                try:
                    return PriceInfo(
                        currency=currency,
                        min_price=float(price) if price else None,
                        max_price=float(price) if price else None,
                        raw_price_text=str(price) if price else None,
                    )
                except Exception:
                    pass

    return PriceInfo()

def _extract_reviews(review_signals: dict, json_ld: list, text_blocks: dict) -> ReviewInfo:
    for item in json_ld:
        if isinstance(item, dict):
            agg = item.get("aggregateRating")
            if isinstance(agg, dict):
                rating = agg.get("ratingValue")
                count = agg.get("reviewCount")
                evidence = ["Found aggregateRating in JSON-LD"]

                if item.get("review"):
                    evidence.append("Found review objects in JSON-LD")

                try:
                    return ReviewInfo(
                        has_reviews=True,
                        review_count=int(count) if count is not None else None,
                        average_rating=float(rating) if rating is not None else None,
                        evidence=evidence,
                    )
                except Exception:
                    return ReviewInfo(
                        has_reviews=True,
                        evidence=evidence,
                    )

    evidence = []
    if review_signals.get("has_reviews_keyword"):
        evidence.append("Page contains review-related text")
    if review_signals.get("has_no_matching_reviews"):
        evidence.append("Page contains 'no matching reviews' text")

    has_reviews = review_signals.get("has_reviews_keyword", False) and not review_signals.get("has_no_matching_reviews", False)

    return ReviewInfo(
        has_reviews=has_reviews,
        evidence=evidence,
    )

def _extract_images(shopify_json: dict | None) -> list[ImageInfo]:
    images = []
    if shopify_json:
        for img in shopify_json.get("data", {}).get("images", []):
            if isinstance(img, str):
                images.append(ImageInfo(url=img))
    return images