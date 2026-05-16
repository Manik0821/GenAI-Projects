from urllib.parse import urlparse, urlunparse, parse_qs


def normalize_product_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    clean = parsed._replace(query="", fragment="", path=path)
    return urlunparse(clean)


def extract_locale_and_handle(url: str) -> tuple[str | None, str]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]

    locale = None
    handle = ""

    if len(parts) >= 3 and parts[1] == "products":
        locale = parts[0]
        handle = parts[2]
    elif len(parts) >= 2 and parts[0] == "products":
        handle = parts[1]
    else:
        raise ValueError(f"Could not extract product handle from URL: {url}")

    return locale, handle


def extract_variant_id(url: str) -> str | None:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    variants = query.get("variant")
    return variants[0] if variants else None


def build_shopify_json_url(url: str) -> str:
    parsed = urlparse(url)
    locale, handle = extract_locale_and_handle(url)

    if locale:
        return f"{parsed.scheme}://{parsed.netloc}/{locale}/products/{handle}.js"
    return f"{parsed.scheme}://{parsed.netloc}/products/{handle}.js"