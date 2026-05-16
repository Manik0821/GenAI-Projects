from bs4 import BeautifulSoup
from app.fetchers import fetch_html

URL = "https://healf.com/products/nbpure-methyl-b12-1-oz"

if __name__ == "__main__":
    result = fetch_html(URL)
    html = result["html"]
    soup = BeautifulSoup(html, "html.parser")

    print("\n=== PAGE TITLE ===")
    print(soup.title.get_text(strip=True) if soup.title else "No <title>")

    print("\n=== MAIN TAGS ===")
    mains = soup.find_all("main")
    print(f"Found {len(mains)} main tags")
    for i, main in enumerate(mains[:3], 1):
        text = main.get_text(" ", strip=True)
        print(f"\n--- main {i} ---")
        print(text[:1000])

    print("\n=== H1 TAGS ===")
    h1s = soup.find_all("h1")
    print(f"Found {len(h1s)} h1 tags")
    for h in h1s[:10]:
        print("-", h.get_text(" ", strip=True))

    print("\n=== H2/H3 SAMPLE ===")
    for h in soup.find_all(["h2", "h3"])[:20]:
        print(f"{h.name}: {h.get_text(' ', strip=True)}")

    print("\n=== IMAGE SAMPLE ===")
    imgs = soup.find_all("img")
    print(f"Found {len(imgs)} img tags")
    for img in imgs[:10]:
        print({
            "src": img.get("src"),
            "alt": img.get("alt")
        })

    print("\n=== KEYWORD CHECKS ===")
    text = soup.get_text(" ", strip=True).lower()
    for keyword in ["ingredients", "suggested use", "review", "methyl b12", "nbpure"]:
        print(keyword, "->", keyword in text)