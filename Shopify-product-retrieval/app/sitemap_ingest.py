import requests
import xml.etree.ElementTree as ET

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HealfProductAgent/0.1)"
}


def fetch_xml(url: str, timeout: int = 20) -> str:
    print(f"[fetch_xml] Fetching: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    print(f"[fetch_xml] Status: {resp.status_code}")
    resp.raise_for_status()
    return resp.text


def parse_xml(xml_text: str):
    return ET.fromstring(xml_text)


def get_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}")[0] + "}"
    return ""
    

def extract_loc_values(xml_text: str) -> list[str]:
    root = parse_xml(xml_text)
    ns = get_namespace(root.tag)
    return [loc.text.strip() for loc in root.findall(f".//{ns}loc") if loc.text]


def discover_products_from_sitemap(base_domain: str = "https://healf.com") -> list[str]:
    sitemap_index_url = f"{base_domain}/sitemap.xml"
    index_xml = fetch_xml(sitemap_index_url)

    child_sitemaps = extract_loc_values(index_xml)
    print(f"[discover_products_from_sitemap] Found {len(child_sitemaps)} child sitemaps")

    all_products = set()

    for sitemap_url in child_sitemaps:
        try:
            print(f"[discover_products_from_sitemap] Checking child sitemap: {sitemap_url}")
            child_xml = fetch_xml(sitemap_url)
            locs = extract_loc_values(child_xml)

            product_urls = [u for u in locs if "/products/" in u]
            print(f"[discover_products_from_sitemap] Product URLs in this sitemap: {len(product_urls)}")

            all_products.update(product_urls)
        except Exception as e:
            print(f"[discover_products_from_sitemap] Failed on {sitemap_url}: {e}")

    all_products = sorted(all_products)
    print(f"[discover_products_from_sitemap] Total unique products: {len(all_products)}")
    return all_products