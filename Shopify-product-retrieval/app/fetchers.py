import requests
from bs4 import BeautifulSoup

from app.url_utils import normalize_product_url, extract_locale_and_handle

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HealfProductAgent/0.1)"
}


def fetch_html(url: str, timeout: int = 20) -> dict:
    clean_url = normalize_product_url(url)
    resp = requests.get(clean_url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return {
        "url": clean_url,
        "status_code": resp.status_code,
        "html": resp.text,
    }


def fetch_shopify_product_json(url: str, timeout: int = 20) -> dict | None:
    parsed_url = normalize_product_url(url)
    locale, handle = extract_locale_and_handle(url)
    base = "https://healf.com"

    candidates = []
    if locale:
        candidates.append(f"{base}/{locale}/products/{handle}.js")
    candidates.append(f"{base}/products/{handle}.js")

    for json_url in candidates:
        try:
            resp = requests.get(json_url, headers=HEADERS, timeout=timeout)
            if resp.status_code != 200:
                continue

            data = resp.json()
            if isinstance(data, dict) and data.get("handle"):
                return {
                    "url": json_url,
                    "status_code": resp.status_code,
                    "data": data,
                }
        except Exception:
            continue

    return None


def make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")