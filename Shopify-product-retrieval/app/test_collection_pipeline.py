from pprint import pprint
from app.collection_ingest import ingest_collection
from app.product_ingest import ingest_products
from app.sitemap_ingest import discover_products_from_sitemap

COLLECTION_URL = "https://healf.com/en-uk/collections/all-products-1"


if __name__ == "__main__":
    collection = ingest_collection(COLLECTION_URL)

    product_urls = collection["product_urls"]
    source = "collection_html"

    if not product_urls and collection["dynamic_collection_suspected"]:
        product_urls = discover_products_from_sitemap("https://healf.com")
        source = "sitemap"

    pprint({
        "collection_url": collection["collection_url"],
        "product_count": len(product_urls),
        "source": source,
        "dynamic_collection_suspected": collection["dynamic_collection_suspected"],
        "sample_urls": product_urls[:5],
    })

    if product_urls:
        products = ingest_products(product_urls[:3])
        pprint([
            {
                "url": p.get("canonical_url", p.get("input_url")),
                "title": p.get("text_blocks", {}).get("title"),
                "has_shopify_json": p.get("shopify_json") is not None if "shopify_json" in p else False,
                "error": p.get("error"),
            }
            for p in products
        ])