from app.ingest import ingest_product_url


def ingest_products(urls: list[str]) -> list[dict]:
    results = []
    for url in urls:
        try:
            results.append(ingest_product_url(url))
        except Exception as e:
            results.append({
                "input_url": url,
                "error": str(e),
            })
    return results