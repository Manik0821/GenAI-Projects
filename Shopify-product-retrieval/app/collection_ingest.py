from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HealfProductAgent/0.1)"
}


def fetch_collection_html(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_product_links_from_collection_html(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/products/" in href:
            full_url = urljoin(base_url, href.split("?")[0])
            links.add(full_url.rstrip("/"))

    return sorted(links)


def ingest_collection(url: str) -> dict:
    html = fetch_collection_html(url)
    links = extract_product_links_from_collection_html(html, url)

    text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True).lower()
    dynamic_collection_suspected = (
        len(links) == 0 and ("loading" in text or "0 results" in text)
    )

    return {
        "collection_url": url,
        "product_urls": links,
        "product_count": len(links),
        "dynamic_collection_suspected": dynamic_collection_suspected,
        "html": html,
    }