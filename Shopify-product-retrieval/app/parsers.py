import json
import re
from bs4 import BeautifulSoup, Tag


SECTION_STOP_HEADINGS = {
    "product description",
    "why nbpure is healf",
    "ingredients",
    "suggested use",
    "you may also like",
    "shop by collection",
    "frequently asked questions",
}


def extract_json_ld(soup: BeautifulSoup) -> list[dict]:
    items = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items.extend([x for x in parsed if isinstance(x, dict)])
            elif isinstance(parsed, dict):
                items.append(parsed)
        except Exception:
            continue
    return items


def extract_text_blocks(soup: BeautifulSoup) -> dict:
    title = _safe_text(soup.find("h1"))
    subheading = _extract_meta_description(soup)

    headings = []
    for h in soup.find_all(["h2", "h3"]):
        text = h.get_text(" ", strip=True)
        if text and text not in headings:
            headings.append(text)

    page_text = soup.get_text("\n", strip=True)

    return {
        "page_text": page_text,
        "title": title,
        "subheading": subheading,
        "headings": headings,
        "sections": extract_named_sections(soup),
    }


def extract_named_sections(soup: BeautifulSoup) -> dict:
    wanted = [
        "Product description",
        "Ingredients",
        "Suggested use",
        "Frequently asked questions",
    ]

    out = {}
    for label in wanted:
        content = extract_section_by_heading(soup, label)
        if content:
            out[label.lower()] = content

    return out


def extract_section_by_heading(soup: BeautifulSoup, heading_text: str) -> str | None:
    target = None
    normalized_target = heading_text.strip().lower()

    for h in soup.find_all(["h2", "h3", "h4"]):
        text = h.get_text(" ", strip=True).lower()
        if text == normalized_target:
            target = h
            break

    if not target:
        return None

    chunks = []
    for sib in target.next_siblings:
        if isinstance(sib, str):
            text = sib.strip()
            if text:
                chunks.append(text)
            continue

        if not isinstance(sib, Tag):
            continue

        if sib.name in ["h2", "h3", "h4"]:
            next_heading = sib.get_text(" ", strip=True).lower()
            if next_heading in SECTION_STOP_HEADINGS:
                break

        text = sib.get_text(" ", strip=True)
        if text:
            chunks.append(text)

    cleaned = " ".join(chunks)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def extract_ingredients_section(page_text: str, sections: dict | None = None) -> list[str]:
    if sections and sections.get("ingredients"):
        return [sections["ingredients"]]

    patterns = [
        r"Ingredients\s*(.*?)(Suggested use|Frequently asked questions|You may also like|Shop by collection|$)",
        r"Supplement facts\s*(.*?)(Suggested use|Frequently asked questions|You may also like|Shop by collection|$)",
    ]

    matches = []
    for pattern in patterns:
        found = re.search(pattern, page_text, re.DOTALL | re.IGNORECASE)
        if found:
            cleaned = " ".join(found.group(1).split())
            if cleaned:
                matches.append(cleaned)

    return matches


def detect_review_signals(page_text: str) -> dict:
    lowered = page_text.lower()
    return {
        "has_reviews_keyword": "review" in lowered or "reviews" in lowered,
        "has_no_matching_reviews": "no matching reviews" in lowered,
    }


def _safe_text(node):
    if not node:
        return None
    text = node.get_text(" ", strip=True)
    return text or None


def _extract_meta_description(soup: BeautifulSoup) -> str | None:
    node = soup.find("meta", attrs={"name": "description"})
    if node and node.get("content"):
        return node["content"].strip()
    return None