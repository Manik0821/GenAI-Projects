from pprint import pprint
from app.ingest import ingest_product_url

URL = "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791"

if __name__ == "__main__":
    data = ingest_product_url(URL)
    pprint({
        "canonical_url": data["canonical_url"],
        "locale": data["locale"],
        "handle": data["handle"],
        "variant_id": data["variant_id"],
        "has_shopify_json": data["shopify_json"] is not None,
        "title": data["text_blocks"]["title"],
        "headings": data["text_blocks"]["headings"][:10],
        "ingredients_sections_found": len(data["ingredients_sections"]),
        "review_signals": data["review_signals"],
        "json_ld_count": len(data["json_ld"]),
    })