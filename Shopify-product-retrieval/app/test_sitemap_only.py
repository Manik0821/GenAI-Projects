from pprint import pprint
from app.sitemap_ingest import discover_products_from_sitemap

if __name__ == "__main__":
    urls = discover_products_from_sitemap("https://healf.com")
    pprint({
        "count": len(urls),
        "sample": urls[:10],
    })