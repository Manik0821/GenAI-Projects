"""Travel Planner — Gradio UI
Carousel → Search / Image Upload → Tabbed place details (mobile-first)
"""

from __future__ import annotations

import base64
import importlib.util
import os
from html import escape
from pathlib import Path

import gradio as gr
import requests

# ── backend modules (hyphenated folder) ────────────────────────────────────
BASE = Path(__file__).parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, BASE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load("places_fetcher", "data-fetcher/places_fetcher.py")
ir = _load("image_recognizer", "data-fetcher/image_recognizer.py")

_WIKIPEDIA_HEADERS = {"User-Agent": "TravelPlanner/1.0 (local development)"}

# ── carousel images ─────────────────────────────────────────────────────────
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".png": "image/png", ".webp": "image/webp", ".avif": "image/avif"}


def _b64(name: str) -> tuple[str, str]:
    """Return (data-uri, mime) or ('', '') if file missing."""
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
        p = BASE / f"{name}{ext}"
        if p.exists():
            mime = _MIME[ext]
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{b64}", mime
    return "", ""


_slides = [s for s in [_b64("travel-1"), _b64("travel-2"), _b64("travel-3")] if s[0]]
_slide_imgs = "".join(f'<img src="{uri}" alt="travel photo" />' for uri, _ in _slides)
_slide_dots = "".join(
    f'<span class="dot{"  active" if i == 0 else ""}" data-idx="{i}"></span>'
    for i in range(len(_slides))
)

CAROUSEL_HTML = f"""
<style>
  .tpc-wrap {{
    position: relative; width: 100%; overflow: hidden;
    border-radius: 0; background: #0f172a; margin: 0;
    min-height: 220px;
  }}
  .tpc-track {{
    display: flex; transition: transform .55s cubic-bezier(.4,0,.2,1);
  }}
  .tpc-track img {{
    min-width: 100%; width: 100%; height: 220px;
    object-fit: cover; flex-shrink: 0; display: block;
  }}
  .tpc-btn {{
    position: absolute; top: 50%; transform: translateY(-50%);
    background: rgba(255,255,255,.25); backdrop-filter: blur(6px);
    border: none; color: #fff; font-size: 1.5rem;
    width: 38px; height: 38px; border-radius: 50%;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background .2s; z-index: 5;
  }}
  .tpc-btn:hover {{ background: rgba(255,255,255,.5); }}
  .tpc-prev {{ left: 10px; }} .tpc-next {{ right: 10px; }}
  /* 30 % translucent gradient overlay — title lives inside it */
  .tpc-title {{
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 30%;
    background: #ffffff90;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    padding-bottom: 26px;
    pointer-events: none;
    z-index: 4;
  }}
  .tpc-title-text {{
    color: #fff;
    font-size: clamp(1.05rem, 3.5vw, 1.6rem);
    font-weight: 800;
    letter-spacing: .5px;
  }}
  .tpc-dots {{
    position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 6px; z-index: 6;
  }}
  .dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(255,255,255,.45); cursor: pointer; transition: background .3s;
  }}
  .dot.active {{ background: #fff; }}
  @media(min-width:640px)  {{ .tpc-track img {{ height: 320px; }} }}
  @media(min-width:1024px) {{ .tpc-track img {{ height: 420px; }} }}
</style>

<div class="tpc-wrap" id="tpc">
  <div class="tpc-track" id="tpc-track">
    {_slide_imgs}
  </div>
  <button class="tpc-btn tpc-prev" id="tpc-prev-btn">&#8249;</button>
  <button class="tpc-btn tpc-next" id="tpc-next-btn">&#8250;</button>
  <div class="tpc-title">
    <span class="tpc-title-text">✈&nbsp; Travel Planner</span>
  </div>
  <div class="tpc-dots" id="tpc-dots">{_slide_dots}</div>
</div>

"""

# ── carousel JS (passed via gr.Blocks js= so it always executes) ─────────────
CAROUSEL_JS = f"""
function() {{
  var cur = 0, total = {len(_slides)}, timer = null;
  function upd(track) {{
    track.style.transform = 'translateX(-' + (cur * 100) + '%)';
    document.querySelectorAll('#tpc-dots .dot').forEach(function(d, i) {{
      d.classList.toggle('active', i === cur);
    }});
  }}
  function init() {{
    var track = document.getElementById('tpc-track');
    if (!track) return false;
    document.getElementById('tpc-prev-btn').addEventListener('click', function() {{
      cur = (cur - 1 + total) % total; upd(track);
    }});
    document.getElementById('tpc-next-btn').addEventListener('click', function() {{
      cur = (cur + 1) % total; upd(track);
    }});
    document.querySelectorAll('#tpc-dots .dot').forEach(function(d) {{
      d.addEventListener('click', function() {{
        cur = parseInt(d.getAttribute('data-idx'), 10); upd(track);
      }});
    }});
    clearInterval(timer);
    timer = setInterval(function() {{ cur = (cur + 1) % total; upd(track); }}, 3000);
    upd(track);
    return true;
  }}
  if (!init()) {{
    var mo = new MutationObserver(function() {{
      if (init()) mo.disconnect();
    }});
    mo.observe(document.documentElement, {{ childList: true, subtree: true }});
  }}
}}
"""

# ── custom CSS ───────────────────────────────────────────────────────────────
CSS = """

main.fillable {
  padding: 0 !important;
}

/* ── remove padding from the Gradio main wrappers called out in DevTools ── */
.tpc-main,
.tpc-main > div,
.padding.svelte-phx28p {
  padding: 0 !important;
  margin: 0 !important;
}

/* ── carousel block: edge-to-edge, no border/shadow, keep overflow visible for overlay ── */
#tpc-carousel-block,
#tpc-carousel-block > div,
#tpc-carousel-block .block,
#tpc-carousel-block .wrap {
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  overflow: visible !important;
}

/* ── hide floating badge/label on the upload accordion ── */
#tpc-upload-acc .label-wrap .icon { display: none !important; }

/* ── hide only the drag/drop placeholder text inside the image input ── */
#tpc-image-input .upload-container span,
#tpc-image-input .upload-container p,
#tpc-image-input .upload-text {
  display: none !important;
}

/* ── search column: full-width stacked with block padding ── */
#tpc-search-col { gap: 10px !important; width: 100% !important; padding: 16px !important; }

/* ── generated data tabs: add inner spacing to tab bodies ── */
#tpc-location-tabs .tabitem,
#tpc-location-tabs .tab-item,
#tpc-location-tabs .tabs > div:last-child,
#tpc-tabs .tabitem,
#tpc-tabs .tab-item,
#tpc-tabs .tabs > div:last-child {
  padding: 14px 16px 20px !important;
}

/* ── pill search input ── */
#tpc-search-input {
  width: 100% !important;
}
#tpc-search-input textarea,
#tpc-search-input input {
  border-radius: 9999px !important;
  padding: 0 22px !important;
  height: 48px !important;
  min-height: 48px !important;
  max-height: 48px !important;
  line-height: 48px !important;
  text-align: left !important;
  display: flex !important;
  align-items: center !important;
  box-shadow: 0 2px 14px rgba(99,102,241,.13) !important;
  border: 1.5px solid #c7d2fe !important;
  font-size: .97rem !important;
  overflow: hidden !important;
  resize: none !important;
}
#tpc-search-input textarea:focus,
#tpc-search-input input:focus {
  border-color: #6366f1 !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important;
  outline: none !important;
}

/* ── full-width capsule buttons ── */
#tpc-search-btn, #tpc-img-btn {
  width: 100% !important;
  border-radius: 9999px !important;
  height: 48px !important;
  min-height: 48px !important;
  padding: 0 28px !important;
  font-size: .97rem !important;
  font-weight: 700 !important;
  letter-spacing: .3px !important;
  transition: box-shadow .2s, transform .15s !important;
}
#tpc-search-btn {
  background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 4px 18px rgba(99,102,241,.38) !important;
}
#tpc-search-btn:hover {
  box-shadow: 0 6px 24px rgba(99,102,241,.55) !important;
  transform: translateY(-1px) !important;
}
#tpc-img-btn {
  background: #fff !important;
  color: #6366f1 !important;
  border: 2px solid #6366f1 !important;
  box-shadow: 0 4px 14px rgba(99,102,241,.16) !important;
}
#tpc-img-btn:hover {
  background: #eef2ff !important;
  box-shadow: 0 6px 20px rgba(99,102,241,.28) !important;
  transform: translateY(-1px) !important;
}

/* ── location summary card — sky-blue travel theme ── */
.tpc-summary {
  background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 55%, #075985 100%);
  color: #fff;
  border-radius: 16px;
  padding: 16px 20px;
  margin: 12px 0 16px;
  font-size: .94rem;
  font-weight: 600;
  line-height: 1.7;
  box-shadow: 0 8px 28px rgba(14,165,233,.35);
}
.tpc-summary strong { font-size: 1.05rem; }
.tpc-summary small  { opacity: .88; font-weight: 400; font-size: .82rem; }
.tpc-location-panel {
  background: #fff;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid #dbeafe;
  box-shadow: 0 8px 24px rgba(14,165,233,.16);
}
.tpc-location-media {
  display: block;
  width: 100%;
  height: 180px;
  object-fit: cover;
  background: #dbeafe;
}
@media(min-width:640px) {
  .tpc-location-media { height: 220px; }
}
.tpc-location-panel-body {
  padding: 14px 16px 16px;
}
.tpc-location-panel-caption {
  margin: 0;
  color: #475569;
  font-size: .9rem;
  line-height: 1.55;
}
.tpc-location-empty {
  padding: 34px 18px;
  text-align: center;
  color: #64748b;
  font-size: .92rem;
  background: linear-gradient(180deg, #f8fbff 0%, #e0f2fe 100%);
}
.tpc-location-facts {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}
.tpc-location-fact {
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.2);
}
.tpc-location-fact-label {
  display: block;
  font-size: .72rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  opacity: .8;
}
.tpc-location-fact-value {
  display: block;
  margin-top: 2px;
  font-size: .9rem;
  font-weight: 700;
}
.tpc-location-summary-line {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid rgba(255,255,255,.2);
  font-size: .84rem;
  opacity: .92;
}

/* ── place card ── */
.tpc-card {
  background: #fff;
  border: 1px solid #e8ecff;
  border-radius: 14px;
  padding: 12px 14px;
  margin-bottom: 10px;
  box-shadow: 0 2px 10px rgba(99,102,241,.09);
  transition: box-shadow .2s, transform .15s;
}
.tpc-card:hover {
  box-shadow: 0 6px 22px rgba(99,102,241,.2);
  transform: translateY(-2px);
}
.tpc-card-name  { margin: 0 0 4px; color: #1d4ed8; font-size: .95rem; font-weight: 700; }
.tpc-card-addr  { margin: 0; color: #64748b; font-size: .82rem; line-height: 1.45; }
.tpc-card-meta  { margin-top: 7px; display: flex; flex-wrap: wrap; gap: 5px; }
.tpc-badge {
  background: #eff6ff; color: #1d4ed8;
  border-radius: 999px; padding: 2px 10px;
  font-size: .73rem; text-decoration: none;
  box-shadow: 0 1px 4px rgba(29,78,216,.1);
}
.tpc-empty { color: #94a3b8; text-align: center; padding: 28px 0; font-size: .9rem; }
"""


# ── helpers ──────────────────────────────────────────────────────────────────
def _cards(items: list[dict], icon: str) -> str:
    if not items:
        return f'<p class="tpc-empty">No {icon} results found nearby.</p>'
    out = []
    for item in items:
        name    = item.get("name", "N/A")
        address = item.get("address", "")
        phone   = item.get("phone", "")
        hours   = item.get("opening_hours", "")
        website = item.get("website", "")
        meta = ""
        if phone:   meta += f'<span class="tpc-badge">📞 {phone}</span>'
        if hours:   meta += f'<span class="tpc-badge">🕐 {hours}</span>'
        if website: meta += f'<a class="tpc-badge" href="{website}" target="_blank">🌐 Website</a>'
        meta_html = f'<div class="tpc-card-meta">{meta}</div>' if meta else ""
        out.append(
            f'<div class="tpc-card">'
            f'<p class="tpc-card-name">{icon} {name}</p>'
            f'<p class="tpc-card-addr">📍 {address}</p>'
            f'{meta_html}</div>'
        )
    return "\n".join(out)


def _joined_place_terms(*parts: object) -> str:
    terms: list[str] = []
    for part in parts:
        value = str(part or "").strip()
        if value and value != "N/A" and value not in terms:
            terms.append(value)
    return ", ".join(terms)


def _wikipedia_image_url(term: str, verify_ssl: bool | str) -> str:
    if not term:
        return ""

    requests_to_try = [
        {
            "action": "query",
            "titles": term,
            "prop": "pageimages",
            "piprop": "original|thumbnail",
            "pithumbsize": 1400,
            "format": "json",
        },
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": term,
            "gsrlimit": 1,
            "prop": "pageimages",
            "piprop": "original|thumbnail",
            "pithumbsize": 1400,
            "format": "json",
        },
    ]

    for params in requests_to_try:
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers=_WIKIPEDIA_HEADERS,
                timeout=12,
                verify=verify_ssl,
            )
            response.raise_for_status()
        except Exception:
            continue

        pages = response.json().get("query", {}).get("pages", {})
        for page in pages.values():
            original = page.get("original", {}).get("source")
            thumbnail = page.get("thumbnail", {}).get("source")
            if original or thumbnail:
                return original or thumbnail
    return ""


def _location_photo_url(place: dict) -> str:
    verify_ssl = getattr(pf, "_SSL_VERIFY", True)
    search_terms: list[str] = []
    candidate_terms = [
        _joined_place_terms(place.get("name")),
        _joined_place_terms(place.get("name"), place.get("country")),
        _joined_place_terms(place.get("name"), place.get("city"), place.get("country")),
        str(place.get("name", "")).strip(),
        _joined_place_terms(place.get("city"), place.get("state"), place.get("country")),
        str(place.get("formatted_address", "")).strip(),
    ]

    for term in candidate_terms:
        if term and term not in search_terms:
            search_terms.append(term)

    for term in search_terms:
        image_url = _wikipedia_image_url(term, verify_ssl)
        if image_url:
            return image_url
    return ""


def _location_map_url(place: dict) -> str:
    api_key = os.getenv("GEOAPIFY_API_KEY", "").strip()
    lat = place.get("lat")
    lon = place.get("lon")
    if not api_key or lat is None or lon is None:
        return ""

    lat = float(lat)
    lon = float(lon)
    return (
        "https://maps.geoapify.com/v1/staticmap"
        f"?style=osm-carto&width=1200&height=420"
        f"&center=lonlat:{lon:.6f},{lat:.6f}"
        "&zoom=13"
        f"&marker=lonlat:{lon:.6f},{lat:.6f};type:material;color:%23ffffff;size:large"
        f"&apiKey={api_key}"
    )


def _location_panel_html(
    *,
    image_url: str,
    alt_text: str,
    caption: str,
    empty_text: str,
    action_url: str = "",
    action_label: str = "",
) -> str:
    if not image_url:
        return (
            '<div class="tpc-location-panel">'
            f'<div class="tpc-location-empty">{escape(empty_text)}</div>'
            '</div>'
        )

    action_html = ""
    if action_url and action_label:
        action_html = (
            '<div class="tpc-card-meta">'
            f'<a class="tpc-badge" href="{escape(action_url, quote=True)}" target="_blank">'
            f'🔗 {escape(action_label)}</a>'
            '</div>'
        )

    return (
        '<div class="tpc-location-panel">'
        f'<img class="tpc-location-media" src="{escape(image_url, quote=True)}" '
        f'alt="{escape(alt_text, quote=True)}" loading="lazy">'
        '<div class="tpc-location-panel-body">'
        f'<p class="tpc-location-panel-caption">{escape(caption)}</p>'
        f'{action_html}'
        '</div>'
        '</div>'
    )


def _location_details_html(place: dict, nearby_summary: str) -> str:
    lat = place.get("lat")
    lon = place.get("lon")
    coordinates = "N/A"
    if lat is not None and lon is not None:
        coordinates = f"{float(lat):.4f}, {float(lon):.4f}"

    facts = [
        ("Address", place.get("formatted_address", "N/A")),
        ("Type", place.get("place_type", "N/A")),
        ("County", place.get("county", "N/A")),
        ("Postcode", place.get("postcode", "N/A")),
        ("Coordinates", coordinates),
    ]
    facts_html = "".join(
        (
            '<div class="tpc-location-fact">'
            f'<span class="tpc-location-fact-label">{escape(label)}</span>'
            f'<span class="tpc-location-fact-value">{escape(str(value))}</span>'
            '</div>'
        )
        for label, value in facts
    )

    return (
        f'<div class="tpc-summary">'
        f'📍 <strong>{escape(str(place.get("name", "N/A")))}</strong>'
        f'&ensp;·&ensp;{escape(str(place.get("city", "N/A")))}, '
        f'{escape(str(place.get("state", "N/A")))}, {escape(str(place.get("country", "N/A")))}<br>'
        f'<small style="opacity:.85">{escape(str(place.get("formatted_address", "N/A")))}'
        f'&ensp;·&ensp;{escape(str(place.get("place_type", "N/A")))}</small>'
        f'<div class="tpc-location-facts">{facts_html}</div>'
        f'<div class="tpc-location-summary-line">{escape(nearby_summary)}</div>'
        f'</div>'
    )


def _explore(place_name: str) -> tuple:
  """Return location tabs plus nearby results."""
  data = None
  err = ""
  try:
    data = pf.explore_place(place_name, radius_meters=5000, limit_per_category=8)
  except Exception as exc:
    err = f'<p style="color:#ef4444;padding:12px">⚠️ {exc}</p>'

  if data is None:
    return (
      gr.update(visible=True),
      gr.update(visible=False),
      err,
      err,
      err,
      err,
      err,
      err,
      err,
      err,
    )

  p = data["place"]
  name = str(p.get("name", "this location"))
  photo_url = _location_photo_url(p)
  map_url = _location_map_url(p)
  map_link = ""
  if p.get("lat") is not None and p.get("lon") is not None:
    map_link = (
      "https://www.google.com/maps/search/?api=1&query="
      f'{p["lat"]},{p["lon"]}'
    )

  location_image_html = _location_panel_html(
    image_url=photo_url,
    alt_text=f"Image of {name}",
    caption=f"Visual preview for {name}.",
    empty_text=f"No preview image was found for {name}.",
  )
  location_details_html = _location_details_html(p, data.get("summary", ""))
  location_map_html = _location_panel_html(
    image_url=map_url,
    alt_text=f"Map of {name}",
    caption=str(p.get("formatted_address", name)),
    empty_text=f"Map preview unavailable for {name}.",
    action_url=map_link,
    action_label="Open in Google Maps",
  )

  icons = {"attractions": "🏛️", "museums": "🖼️",
       "restaurants": "🍽️", "hotels": "🏨", "parks": "🌿"}
  tabs = tuple(_cards(data.get(k, []), v) for k, v in icons.items())
  has_nearby_results = any(bool(data.get(k)) for k in icons)
  return (
    gr.update(visible=True),
    gr.update(visible=has_nearby_results),
    location_image_html,
    location_details_html,
    location_map_html,
  ) + tabs


# ── event handlers ────────────────────────────────────────────────────────────
BLANK = (
  gr.update(visible=False),
  gr.update(visible=False),
  "",
  "",
  "",
  "",
  "",
  "",
  "",
  "",
)


def on_text_search(place: str):
    if not place.strip():
        return BLANK
    return _explore(place.strip())


def on_image_search(image_path):
    if not image_path:
        return ("",) + BLANK
    detected = ir.recognize_place_from_image(image_path)
    return (detected,) + _explore(detected)


# ── UI ────────────────────────────────────────────────────────────────────────
_TABS = [
    ("🏛️ Attractions",  "attractions"),
    ("🖼️ Museums",      "museums"),
    ("🍽️ Restaurants",  "restaurants"),
    ("🏨 Hotels",       "hotels"),
    ("🌿 Parks",        "parks"),
]

with gr.Blocks(css=CSS, js=CAROUSEL_JS, theme=gr.themes.Soft(), title="Travel Planner") as demo:

    # ── carousel ──
  gr.HTML(
    CAROUSEL_HTML,
    elem_id="tpc-carousel-block",
    elem_classes=["tpc-component", "tpc-carousel-block"],
  )

  with gr.Column(elem_classes=["tpc-component", "tpc-main"]):
    with gr.Column(elem_classes=["tpc-component", "tpc-content"]):

      # ── text search: input above button (column) ──
      with gr.Column(
        elem_id="tpc-search-col",
        elem_classes=["tpc-component", "tpc-search-col"],
      ):
        txt = gr.Textbox(
          placeholder="🔍  Paris, Mumbai, Taj Mahal, Goa…",
          label="Search by Place Name",
          show_label=False,
          container=False,
          elem_id="tpc-search-input",
          elem_classes=["tpc-component", "tpc-search-input"],
        )
        search_btn = gr.Button(
          "Explore ✈", variant="primary",
          elem_id="tpc-search-btn",
          elem_classes=["tpc-component", "tpc-search-btn"],
        )

      # ── image upload ──
      with gr.Accordion(
        "📷  Detect place from a photo",
        open=False,
        elem_id="tpc-upload-acc",
        elem_classes=["tpc-component", "tpc-upload-acc"],
      ):
        with gr.Row(elem_classes=["tpc-component", "tpc-upload-row"]):
          img_in = gr.Image(
            type="filepath",
            label="",
            show_label=False,
            height=190,
            scale=3,
            elem_id="tpc-image-input",
            elem_classes=["tpc-component", "tpc-image-input"],
          )
          with gr.Column(
            scale=2,
            min_width=160,
            elem_classes=["tpc-component", "tpc-detected-col"],
          ):
            detected = gr.Textbox(
              label="AI Detected Place",
              interactive=False,
              lines=2,
              elem_classes=["tpc-component", "tpc-detected-output"],
            )
            img_btn = gr.Button(
              "Detect & Explore", variant="secondary",
              elem_id="tpc-img-btn",
              elem_classes=["tpc-component", "tpc-img-btn"],
            )

      # ── current location tabs ──
      with gr.Column(
        visible=False,
        elem_classes=["tpc-component", "tpc-location-section"],
      ) as location_section:
        with gr.Tabs(
          elem_id="tpc-location-tabs",
          elem_classes=["tpc-component", "tpc-location-tabs"],
        ):
          with gr.Tab("🖼️ Image", elem_classes=["tpc-component", "tpc-location-tab"]):
            location_image_out = gr.HTML(
              elem_classes=["tpc-component", "tpc-location-output"]
            )
          with gr.Tab("📍 Details", elem_classes=["tpc-component", "tpc-location-tab"]):
            location_details_out = gr.HTML(
              elem_classes=["tpc-component", "tpc-location-output"]
            )
          with gr.Tab("🗺️ Map", elem_classes=["tpc-component", "tpc-location-tab"]):
            location_map_out = gr.HTML(
              elem_classes=["tpc-component", "tpc-location-output"]
            )

      # ── tabs ──
      tab_outputs = []
      with gr.Column(
        visible=False,
        elem_classes=["tpc-component", "tpc-results-section"],
      ) as results_section:
        with gr.Tabs(
          elem_id="tpc-tabs",
          elem_classes=["tpc-component", "tpc-tabs"],
        ):
          for label, _ in _TABS:
            with gr.Tab(label, elem_classes=["tpc-component", "tpc-tab"]):
              tab_outputs.append(
                gr.HTML(elem_classes=["tpc-component", "tpc-tab-output"])
              )

    # ── wire events ──
    text_outs = [
      location_section,
      results_section,
      location_image_out,
      location_details_out,
      location_map_out,
    ] + tab_outputs

    search_btn.click(fn=on_text_search, inputs=txt, outputs=text_outs)
    txt.submit(fn=on_text_search, inputs=txt, outputs=text_outs)

    img_btn.click(
        fn=on_image_search,
        inputs=img_in,
        outputs=[detected] + text_outs,
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False, show_error=True)
