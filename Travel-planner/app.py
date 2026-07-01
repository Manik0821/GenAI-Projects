"""Travel Planner - Gradio UI.

Carousel -> Search / Image Upload -> Tabbed place details (mobile-first).
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import re
from html import escape
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import requests

# â”€â”€ backend modules (hyphenated folder) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE = Path(__file__).parent


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, BASE / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load("places_fetcher", "data-fetcher/places_fetcher.py")
ir = _load("image_recognizer", "data-fetcher/image_recognizer.py")

_WIKIPEDIA_HEADERS = {"User-Agent": "TravelPlanner/1.0 (local development)"}

_NATURAL_LANGUAGE_QUERY_RE = re.compile(
  r"^(?:where(?:\s+exactly)?\s+is|where's|what\s+is|which\s+place\s+is|"
  r"tell\s+me\s+about|show\s+me|find|locate|search\s+for|take\s+me\s+to|"
  r"directions\s+to|how\s+do\s+i\s+get\s+to|"
  r"i\s+(?:want|would\s+like)\s+to\s+(?:visit|see|go\s+to|explore))\b",
  re.IGNORECASE,
)
_QUERY_PREFIX_RE = re.compile(
  r"^(?:where(?:\s+exactly)?\s+is|where's|what\s+is|which\s+place\s+is|"
  r"tell\s+me\s+about|show\s+me|find|locate|search\s+for|take\s+me\s+to|"
  r"directions\s+to|how\s+do\s+i\s+get\s+to|"
  r"i\s+(?:want|would\s+like)\s+to\s+(?:visit|see|go\s+to|explore))\s+",
  re.IGNORECASE,
)

ICON_PLANE = "\u2708"
ICON_SEARCH = "\U0001F50D"
ICON_CAMERA = "\U0001F4F7"
ICON_IMAGE = "\U0001F5BC\ufe0f"
ICON_PIN = "\U0001F4CD"
ICON_MAP = "\U0001F5FA\ufe0f"
ICON_ATTRACTIONS = "\U0001F3DB\ufe0f"
ICON_MUSEUM = "\U0001F5BC\ufe0f"
ICON_RESTAURANT = "\U0001F37D\ufe0f"
ICON_HOTEL = "\U0001F3E8"
ICON_PARK = "\U0001F33F"
ICON_CITY = "\U0001F3D9\ufe0f"
ICON_PHONE = "\U0001F4DE"
ICON_CLOCK = "\U0001F550"
ICON_GLOBE = "\U0001F310"
ICON_LINK = "\U0001F517"
ICON_WARNING = "\u26A0\ufe0f"
RECENT_SEARCH_COOKIE = "travel_planner_recent_searches"
RECENT_SEARCH_LIMIT = 8

# â”€â”€ carousel images â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
     ".png": "image/png", ".webp": "image/webp", ".avif": "image/avif"}


def _static_file_url(path: Path) -> str:
  prefix = os.getenv("TRAVEL_APP_ROUTE_PREFIX", "").strip()
  if prefix:
    return f"{prefix.rstrip('/')}/gradio_api/file=" + quote(path.resolve().as_posix(), safe="/")
  return "gradio_api/file=" + quote(path.resolve().as_posix(), safe="/")


def _hero_asset(name: str) -> str:
  for ext in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
    p = BASE / f"{name}{ext}"
    if p.exists():
      return _static_file_url(p)
  return ""


def _normalize_search_text(value: str) -> str:
  return re.sub(r"\s+", " ", str(value or "")).strip()


def _needs_llm_query_resolution(query: str) -> bool:
  normalized = _normalize_search_text(query)
  if not normalized:
    return False
  return "?" in normalized or bool(_NATURAL_LANGUAGE_QUERY_RE.match(normalized))


def _heuristic_place_query(query: str) -> str:
  candidate = _normalize_search_text(query)
  if not candidate:
    return ""
  candidate = candidate.strip(" \"'")
  candidate = re.sub(r"[?!.,]+$", "", candidate).strip()
  candidate = _QUERY_PREFIX_RE.sub("", candidate).strip(" ,")
  candidate = re.sub(r"^(?:the\s+location\s+of|location\s+of)\s+", "", candidate, flags=re.IGNORECASE)
  candidate = re.sub(r"\s+(?:please|for\s+me)$", "", candidate, flags=re.IGNORECASE)
  return candidate or _normalize_search_text(query)


@functools.lru_cache(maxsize=1)
def _llm_query_module():
  return _load("ai_model", "Model-setup/ai_model.py")


@functools.lru_cache(maxsize=128)
def _resolve_place_query(query: str) -> str:
  normalized = _normalize_search_text(query)
  if not normalized:
    return ""

  fallback = _heuristic_place_query(normalized)
  if not _needs_llm_query_resolution(normalized):
    return fallback

  prompt = (
    "Extract the single place the user wants to search for in a travel app. "
    "Return ONLY the canonical place name. Include city, state, or country only when it helps disambiguate. "
    "Do not answer the question. Do not add bullets, labels, or explanation. "
    "If there is no identifiable place, reply UNKNOWN.\n\n"
    f"User query: {normalized}\n"
    f"Fallback guess: {fallback or 'UNKNOWN'}"
  )

  try:
    response = _llm_query_module().ask_with_failover(
      prompt,
      system_prompt="You extract place names from travel search queries.",
      temperature=0.0,
      max_tokens=48,
    )
  except Exception:
    return fallback

  candidate = _normalize_search_text(response.splitlines()[0] if response else "")
  candidate = candidate.strip(" \"'")
  candidate = re.sub(r"^(?:place|location)\s*:\s*", "", candidate, flags=re.IGNORECASE)
  candidate = re.sub(r"[?!.,]+$", "", candidate).strip()
  if not candidate or candidate.lower() == "unknown":
    return fallback
  return candidate


_slides = [s for s in [_hero_asset("travel-1"), _hero_asset("travel-2"), _hero_asset("travel-3"), _hero_asset("travel-4")] if s]
_slide_imgs = "".join(
  (
    f'<img src="{escape(uri, quote=True)}" alt="travel photo {idx + 1}" '
    f'loading="{"eager" if idx == 0 else "lazy"}" '
    f'fetchpriority="{"high" if idx == 0 else "low"}" decoding="async" />'
  )
  for idx, uri in enumerate(_slides)
)
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
    display: flex; transition: transform .9s cubic-bezier(.22,1,.36,1);
    will-change: transform;
  }}
  .tpc-track img {{
    min-width: 100%; width: 100%; height: 220px;
    object-fit: cover; object-position: center 24%; flex-shrink: 0; display: block;
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
  /* 30% translucent gradient overlay - title lives inside it. */
  .tpc-title {{
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 24%;
    background: #ffffffa0;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    padding-bottom: 18px;
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
  @media(min-width:1024px) {{
    .tpc-track img {{ height: 460px; object-position: center 26%; }}
    .tpc-title {{ height: 22%; padding-bottom: 16px; }}
  }}
</style>

<div class="tpc-wrap" id="tpc">
  <div class="tpc-track" id="tpc-track">
    {_slide_imgs}
  </div>
  <button class="tpc-btn tpc-prev" id="tpc-prev-btn">&#8249;</button>
  <button class="tpc-btn tpc-next" id="tpc-next-btn">&#8250;</button>
  <div class="tpc-title">
    <span class="tpc-title-text">{ICON_PLANE}&nbsp; Travel Planner</span>
  </div>
  <div class="tpc-dots" id="tpc-dots">{_slide_dots}</div>
</div>

"""

# â”€â”€ carousel JS (passed via gr.Blocks js= so it always executes) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CAROUSEL_JS = f"""
function() {{
  var cur = 0, total = {len(_slides)}, timer = null;
  var recentCookieName = '{RECENT_SEARCH_COOKIE}';
  var recentSearchLimit = {RECENT_SEARCH_LIMIT};
  function upd(track) {{
    track.style.transform = 'translateX(-' + (cur * 100) + '%)';
    document.querySelectorAll('#tpc-dots .dot').forEach(function(d, i) {{
      d.classList.toggle('active', i === cur);
    }});
  }}
  function getWrappedButton(elemId) {{
    var wrapper = document.getElementById(elemId);
    if (!wrapper) return null;
    return wrapper.querySelector('button') || wrapper;
  }}
  function getTextboxInput(elemId) {{
    var wrapper = document.getElementById(elemId);
    if (!wrapper) return null;
    return wrapper.querySelector('textarea, input');
  }}
  function readTextboxValue(elemId) {{
    var input = getTextboxInput(elemId);
    return input ? input.value : '';
  }}
  function escapeHtml(value) {{
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }}
  function getCookie(name) {{
    var prefix = name + '=';
    var parts = document.cookie ? document.cookie.split(';') : [];
    for (var i = 0; i < parts.length; i += 1) {{
      var cookie = parts[i].trim();
      if (cookie.indexOf(prefix) === 0) return decodeURIComponent(cookie.slice(prefix.length));
    }}
    return '';
  }}
  function setCookie(name, value, days) {{
    var maxAge = Math.max(1, Math.floor(days * 24 * 60 * 60));
    document.cookie = name + '=' + encodeURIComponent(value) + '; path=/; max-age=' + maxAge + '; SameSite=Lax';
  }}
  function normalizeSearchValue(value) {{
    return String(value || '').trim().replace(/\\s+/g, ' ');
  }}
  function loadRecentSearches() {{
    var raw = getCookie(recentCookieName);
    if (!raw) return [];
    try {{
      var parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .map(normalizeSearchValue)
        .filter(Boolean)
        .slice(0, recentSearchLimit);
    }} catch (err) {{
      return [];
    }}
  }}
  function saveRecentSearches(items) {{
    setCookie(recentCookieName, JSON.stringify(items.slice(0, recentSearchLimit)), 365);
  }}
  function recordRecentSearch(value) {{
    var normalized = normalizeSearchValue(value);
    if (!normalized) return false;
    var items = loadRecentSearches().filter(function(item) {{
      return item.toLowerCase() !== normalized.toLowerCase();
    }});
    items.unshift(normalized);
    saveRecentSearches(items);
    return true;
  }}
  function updateRecentNavState() {{
    var shell = document.getElementById('tpc-recent-shell');
    var viewport = document.getElementById('tpc-recent-viewport');
    var prev = document.getElementById('tpc-recent-prev');
    var next = document.getElementById('tpc-recent-next');
    if (!shell || !viewport || !prev || !next) return;
    var hasOverflow = viewport.scrollWidth - viewport.clientWidth > 8;
    shell.classList.toggle('has-overflow', hasOverflow);
    prev.disabled = !hasOverflow || viewport.scrollLeft <= 4;
    next.disabled = !hasOverflow || viewport.scrollLeft + viewport.clientWidth >= viewport.scrollWidth - 4;
  }}
  function triggerSearch(value) {{
    syncTextbox('tpc-search-input', value);
    var button = getWrappedButton('tpc-search-btn');
    if (button) button.click();
  }}
  function renderRecentSearches() {{
    var shell = document.getElementById('tpc-recent-shell');
    var track = document.getElementById('tpc-recent-track');
    var viewport = document.getElementById('tpc-recent-viewport');
    if (!shell || !track || !viewport) return false;
    var items = loadRecentSearches();
    track.innerHTML = '';
    if (!items.length) {{
      shell.hidden = true;
      shell.classList.remove('is-visible');
      updateRecentNavState();
      return true;
    }}
    items.forEach(function(item) {{
      var card = document.createElement('button');
      card.type = 'button';
      card.className = 'tpc-recent-card';
      card.innerHTML = '<strong>' + escapeHtml(item) + '</strong><span>Search again</span>';
      card.addEventListener('click', function() {{
        recordRecentSearch(item);
        renderRecentSearches();
        triggerSearch(item);
      }});
      track.appendChild(card);
    }});
    viewport.scrollLeft = 0;
    shell.hidden = false;
    shell.classList.add('is-visible');
    updateRecentNavState();
    return true;
  }}
  function scrollRecentSearches(direction) {{
    var viewport = document.getElementById('tpc-recent-viewport');
    if (!viewport) return;
    var amount = Math.max(220, Math.floor(viewport.clientWidth * 0.82));
    viewport.scrollBy({{ left: direction * amount, behavior: 'smooth' }});
    window.setTimeout(updateRecentNavState, 260);
  }}
  function bindRecentSearchControls() {{
    var searchButton = getWrappedButton('tpc-search-btn');
    var searchInput = getTextboxInput('tpc-search-input');
    if (searchButton && searchButton.dataset.recentBound !== '1') {{
      searchButton.dataset.recentBound = '1';
      searchButton.addEventListener('click', function() {{
        if (recordRecentSearch(readTextboxValue('tpc-search-input'))) renderRecentSearches();
      }});
    }}
    if (searchInput && searchInput.dataset.recentBound !== '1') {{
      searchInput.dataset.recentBound = '1';
      searchInput.addEventListener('keydown', function(evt) {{
        if (evt.key === 'Enter' && !evt.shiftKey) {{
          if (recordRecentSearch(searchInput.value)) renderRecentSearches();
        }}
      }});
    }}
  }}
  function initRecentSearches() {{
    var shell = document.getElementById('tpc-recent-shell');
    var viewport = document.getElementById('tpc-recent-viewport');
    var prev = document.getElementById('tpc-recent-prev');
    var next = document.getElementById('tpc-recent-next');
    if (!shell || !viewport || !prev || !next) return false;
    if (shell.dataset.bound !== '1') {{
      shell.dataset.bound = '1';
      prev.addEventListener('click', function() {{ scrollRecentSearches(-1); }});
      next.addEventListener('click', function() {{ scrollRecentSearches(1); }});
      viewport.addEventListener('scroll', updateRecentNavState);
      window.addEventListener('resize', updateRecentNavState);
    }}
    bindRecentSearchControls();
    renderRecentSearches();
    return true;
  }}
  function init() {{
    var root = document.getElementById('tpc');
    var track = document.getElementById('tpc-track');
    if (!root || !track) return false;
    if (root.dataset.bound === '1') return true;
    root.dataset.bound = '1';
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
    timer = setInterval(function() {{ cur = (cur + 1) % total; upd(track); }}, 5000);
    upd(track);
    return true;
  }}
  function syncTextbox(elemId, value) {{
    var wrapper = document.getElementById(elemId);
    if (!wrapper) return;
    var input = wrapper.querySelector('textarea, input');
    if (!input) return;
    var proto = input.tagName === 'TEXTAREA'
      ? window.HTMLTextAreaElement.prototype
      : window.HTMLInputElement.prototype;
    var descriptor = Object.getOwnPropertyDescriptor(proto, 'value');
    if (descriptor && descriptor.set) {{
      descriptor.set.call(input, value);
    }} else {{
      input.value = value;
    }}
    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
  }}
  function resetUploadState() {{
    var preview = document.getElementById('tpc-client-upload-preview');
    var empty = document.getElementById('tpc-client-upload-empty');
    var note = document.getElementById('tpc-client-upload-note');
    window.__tpcPreparedImage = '';
    if (preview) {{
      preview.removeAttribute('src');
      preview.style.display = 'none';
    }}
    if (empty) empty.style.display = 'flex';
    if (note) note.textContent = 'No image selected yet.';
    syncTextbox('tpc-image-data', '');
  }}
  function prepareImageData(file, onDone, onError) {{
    var reader = new FileReader();
    reader.onload = function(evt) {{
      var img = new Image();
      img.onload = function() {{
        var maxSide = 1600;
        var width = img.naturalWidth || img.width;
        var height = img.naturalHeight || img.height;
        if (!width || !height) {{
          onError('Could not read that image.');
          return;
        }}
        if (width > maxSide || height > maxSide) {{
          var scale = Math.min(maxSide / width, maxSide / height);
          width = Math.max(1, Math.round(width * scale));
          height = Math.max(1, Math.round(height * scale));
        }}
        var canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        var ctx = canvas.getContext('2d');
        if (!ctx) {{
          onError('Canvas is unavailable in this browser.');
          return;
        }}
        ctx.drawImage(img, 0, 0, width, height);
        onDone(canvas.toDataURL('image/jpeg', 0.88));
      }};
      img.onerror = function() {{ onError('That file is not a supported image.'); }};
      img.src = evt.target && evt.target.result ? evt.target.result : '';
    }};
    reader.onerror = function() {{ onError('Could not read that file.'); }};
    reader.readAsDataURL(file);
  }}
  function initUploader() {{
    var input = document.getElementById('tpc-file-input');
    if (!input) return false;
    if (input.dataset.bound === '1') return true;
    input.dataset.bound = '1';

    var preview = document.getElementById('tpc-client-upload-preview');
    var empty = document.getElementById('tpc-client-upload-empty');
    var note = document.getElementById('tpc-client-upload-note');
    var detectWrapper = document.getElementById('tpc-img-btn');
    var detectButton = detectWrapper ? (detectWrapper.querySelector('button') || detectWrapper) : null;

    resetUploadState();

    if (detectButton && detectButton.dataset.bound !== '1') {{
      detectButton.dataset.bound = '1';
      detectButton.addEventListener('click', function() {{
        syncTextbox('tpc-image-data', window.__tpcPreparedImage || '');
      }});
    }}

    input.addEventListener('change', function() {{
      var file = input.files && input.files[0];
      if (!file) {{
        resetUploadState();
        return;
      }}
      if (note) note.textContent = 'Preparing ' + file.name + '...';
      prepareImageData(file, function(dataUrl) {{
        window.__tpcPreparedImage = dataUrl;
        syncTextbox('tpc-image-data', dataUrl);
        if (preview) {{
          preview.src = dataUrl;
          preview.style.display = 'block';
        }}
        if (empty) empty.style.display = 'none';
        if (note) note.textContent = file.name + ' is ready for detection.';
      }}, function(message) {{
        resetUploadState();
        if (note) note.textContent = message;
      }});
    }});
    return true;
  }}
  if (!init()) {{
    var mo = new MutationObserver(function() {{
      var ready = init();
      var uploadReady = initUploader();
      var recentReady = initRecentSearches();
      if (ready && uploadReady && recentReady) mo.disconnect();
    }});
    mo.observe(document.documentElement, {{ childList: true, subtree: true }});
  }}
  initUploader();
  initRecentSearches();
}}
"""

RECENT_SEARCHES_HTML = """
<div class="tpc-recent-shell" id="tpc-recent-shell" hidden>
  <div class="tpc-recent-head">
    <div>
      <p class="tpc-recent-kicker">Recent Searches</p>
      <h3 class="tpc-recent-title">Jump back into a place you searched before</h3>
    </div>
    <div class="tpc-recent-nav">
      <button type="button" class="tpc-recent-btn" id="tpc-recent-prev" aria-label="Scroll recent searches left">&#8249;</button>
      <button type="button" class="tpc-recent-btn" id="tpc-recent-next" aria-label="Scroll recent searches right">&#8250;</button>
    </div>
  </div>
  <div class="tpc-recent-viewport" id="tpc-recent-viewport">
    <div class="tpc-recent-track" id="tpc-recent-track"></div>
  </div>
</div>
"""

UPLOAD_HTML = """
<div class="tpc-client-upload-shell">
  <label class="tpc-client-upload-box" for="tpc-file-input">
    <input id="tpc-file-input" class="tpc-client-upload-input" type="file" accept="image/*" />
    <img id="tpc-client-upload-preview" class="tpc-client-upload-preview" alt="Selected place preview" />
    <div id="tpc-client-upload-empty" class="tpc-client-upload-empty">
      <span class="tpc-client-upload-badge">Upload Photo</span>
      <strong>Upload an Image</strong>
      <p>JPG, PNG, WEBP and GIF work well. The image is compressed in your browser before it is sent.</p>
    </div>
  </label>
  <p id="tpc-client-upload-note" class="tpc-client-upload-note">No image selected yet.</p>
</div>
"""

# â”€â”€ custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CSS = """

:root,
html,
body,
.gradio-container,
.gradio-container > .main,
main.fillable {
  --body-background-fill: #f4faff !important;
  --body-background-fill-dark: #f4faff !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #f8fbff !important;
  --block-background-fill: #ffffff !important;
  --body-text-color: #0f172a !important;
  --body-text-color-subdued: #475569 !important;
  background: linear-gradient(180deg, #f4faff 0%, #edf6ff 52%, #ffffff 100%) !important;
  color: #0f172a !important;
  color-scheme: light !important;
}

html,
body {
  margin: 0 !important;
  min-height: 100% !important;
}

main.fillable {
  padding: 0 !important;
}

/* â”€â”€ remove padding from the Gradio main wrappers called out in DevTools â”€â”€ */
.tpc-main,
.tpc-main > div,
.padding.svelte-phx28p {
  padding: 0 !important;
  margin: 0 !important;
}

/* â”€â”€ carousel block: edge-to-edge, no border/shadow, keep overflow visible for overlay â”€â”€ */
#tpc-carousel-block,
#tpc-carousel-block > div,
#tpc-carousel-block .block,
#tpc-carousel-block .wrap {
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  overflow: visible !important;
}

/* â”€â”€ hide floating badge/label on the upload accordion â”€â”€ */
#tpc-upload-acc .label-wrap .icon { display: none !important; }

/* â”€â”€ client-side image upload card â”€â”€ */
#tpc-image-input {
  width: 100% !important;
}
.tpc-client-upload-shell {
  width: 100%;
}
.tpc-client-upload-box {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 190px;
  border-radius: 18px;
  border: 1.5px dashed #93c5fd;
  background: linear-gradient(180deg, #f8fbff 0%, #e0f2fe 100%);
  overflow: hidden;
  cursor: pointer;
  transition: border-color .2s, box-shadow .2s, transform .15s;
}
.tpc-client-upload-box:hover {
  border-color: #6366f1;
  box-shadow: 0 8px 24px rgba(99,102,241,.18);
  transform: translateY(-1px);
}
.tpc-client-upload-input {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}
.tpc-client-upload-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  padding: 22px;
  text-align: center;
  color: #334155;
}
.tpc-client-upload-empty strong {
  display: block;
  width: 100%;
  color: #0f172a;
  font-size: .98rem;
  text-align: center;
}
.tpc-client-upload-empty p {
  margin: 0;
  width: 100%;
  font-size: .83rem;
  line-height: 1.45;
  color: #64748b;
  text-align: center;
}
.tpc-client-upload-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 5px 11px;
  border-radius: 999px;
  background: rgba(99,102,241,.12);
  color: #4338ca;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.tpc-client-upload-preview {
  display: none;
  width: 100%;
  height: 190px;
  object-fit: cover;
}
.tpc-client-upload-note {
  margin: 10px 4px 0;
  color: #64748b;
  font-size: .82rem;
  line-height: 1.4;
  text-align: center;
}

#tpc-image-data {
  position: absolute !important;
  left: -10000px !important;
  top: 0 !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  overflow: hidden !important;
  pointer-events: none !important;
}
#tpc-image-data textarea,
#tpc-image-data input {
  min-height: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  border: 0 !important;
}

/* â”€â”€ search column: full-width stacked with block padding â”€â”€ */
#tpc-search-col { gap: 10px !important; width: 100% !important; padding: 16px !important; }

/* â”€â”€ recent searches carousel â”€â”€ */
#tpc-recent-searches {
  width: 100% !important;
}
.tpc-recent-shell {
  display: none;
  width: 100%;
  gap: 12px;
  padding: 14px 16px 2px;
  border-radius: 20px;
  border: 1px solid #dbeafe;
  background: linear-gradient(180deg, rgba(255,255,255,.96) 0%, rgba(239,246,255,.92) 100%);
  box-shadow: 0 10px 30px rgba(14,165,233,.12);
}
.tpc-recent-shell.is-visible {
  display: flex;
  flex-direction: column;
}
.tpc-recent-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.tpc-recent-kicker {
  margin: 0 0 4px;
  color: #0284c7;
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
}
.tpc-recent-title {
  margin: 0;
  color: #0f172a;
  font-size: .96rem;
  line-height: 1.35;
}
.tpc-recent-nav {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}
.tpc-recent-btn {
  width: 34px;
  height: 34px;
  border-radius: 999px;
  border: 1px solid #bfdbfe;
  background: #ffffff;
  color: #2563eb;
  font-size: 1rem;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 14px rgba(37,99,235,.12);
  transition: transform .15s, box-shadow .2s, background .2s;
}
.tpc-recent-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(37,99,235,.18);
  background: #eff6ff;
}
.tpc-recent-btn:disabled {
  opacity: .45;
  cursor: default;
  box-shadow: none;
}
.tpc-recent-viewport {
  overflow-x: auto;
  padding-bottom: 10px;
  scroll-behavior: smooth;
  scrollbar-width: none;
  -ms-overflow-style: none;
}
.tpc-recent-viewport::-webkit-scrollbar {
  display: none;
}
.tpc-recent-track {
  display: flex;
  gap: 12px;
}
.tpc-recent-card {
  flex: 0 0 84%;
  max-width: 84%;
  min-height: 112px;
  padding: 16px;
  border-radius: 18px;
  border: 1px solid #bfdbfe;
  background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
  color: #0f172a;
  text-align: left;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 14px;
  box-shadow: 0 8px 20px rgba(59,130,246,.12);
  transition: transform .16s, box-shadow .2s, border-color .2s;
}
.tpc-recent-card:hover {
  transform: translateY(-2px);
  border-color: #60a5fa;
  box-shadow: 0 12px 26px rgba(59,130,246,.2);
}
.tpc-recent-card strong {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  color: #0f172a;
  font-size: .95rem;
  line-height: 1.45;
}
.tpc-recent-card span {
  color: #2563eb;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .02em;
}
@media(max-width:639px) {
  .tpc-recent-shell {
    padding: 14px 14px 2px;
  }
  .tpc-recent-head {
    align-items: center;
  }
  .tpc-recent-title {
    font-size: .9rem;
  }
  .tpc-recent-card {
    flex-basis: 84%;
    max-width: 84%;
    min-height: 102px;
    padding: 14px;
  }
}
@media(min-width:640px) and (max-width:1023px) {
  .tpc-recent-card {
    flex-basis: 50%;
    max-width: 50%;
  }
}
@media(min-width:1024px) {
  .tpc-recent-card {
    flex-basis: 30%;
    max-width: 30%;
  }
}

/* â”€â”€ generated data tabs: add inner spacing to tab bodies â”€â”€ */
#tpc-location-tabs .tabitem,
#tpc-location-tabs .tab-item,
#tpc-location-tabs .tabs > div:last-child,
#tpc-tabs .tabitem,
#tpc-tabs .tab-item,
#tpc-tabs .tabs > div:last-child {
  padding: 14px 16px 20px !important;
}

/* â”€â”€ pill search input â”€â”€ */
#tpc-search-input {
  width: 100% !important;
  border-radius: 9999px !important;
  overflow: hidden !important;
  background: #ffffff !important;
}
#tpc-search-input > div,
#tpc-search-input .wrap,
#tpc-search-input .scroll-hide,
#tpc-search-input label {
  border-radius: 9999px !important;
  overflow: hidden !important;
  background: #ffffff !important;
}
#tpc-search-input textarea,
#tpc-search-input input {
  border-radius: 9999px !important;
  background: #ffffff !important;
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

/* â”€â”€ full-width capsule buttons â”€â”€ */
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

/* â”€â”€ location summary card â€” sky-blue travel theme â”€â”€ */
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
.tpc-location-slide-input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}
.tpc-location-slider-shell {
  position: relative;
  overflow: hidden;
  background: #dbeafe;
}
.tpc-location-slider-track {
  display: flex;
  transition: transform .55s cubic-bezier(.22, 1, .36, 1);
  will-change: transform;
}
.tpc-location-slide {
  flex: 0 0 100%;
  min-width: 100%;
}
.tpc-location-media {
  display: block;
  width: 100%;
  height: 180px;
  object-fit: contain;
  object-position: center center;
  background: #dbeafe;
}
@media(min-width:640px) {
  .tpc-location-media { height: 260px; }
}
@media(min-width:1024px) {
  .tpc-location-media { height: 320px; }
}
.tpc-location-slider-controls {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.tpc-location-slider-btn {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 38px;
  height: 38px;
  border-radius: 999px;
  background: rgba(15, 23, 42, .34);
  color: #fff;
  font-size: 1.25rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 8px 18px rgba(15, 23, 42, .22);
  cursor: pointer;
  opacity: 0;
  pointer-events: none;
  transition: opacity .2s, background .2s, transform .15s;
}
.tpc-location-slider-btn:hover {
  background: rgba(15, 23, 42, .52);
}
.tpc-location-slider-btn.prev {
  left: 14px;
}
.tpc-location-slider-btn.next {
  right: 14px;
}
.tpc-location-slider-dots {
  position: absolute;
  left: 50%;
  bottom: 14px;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  z-index: 2;
}
.tpc-location-slider-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,.48);
  box-shadow: 0 2px 8px rgba(15, 23, 42, .2);
  cursor: pointer;
  transition: transform .15s, background .2s;
}
.tpc-location-slider-dot:hover {
  transform: scale(1.08);
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

/* â”€â”€ place card â”€â”€ */
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


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _cards(items: list[dict], icon: str, empty_text: str | None = None) -> str:
  if not items:
    message = empty_text or f"No {icon} results found nearby."
    return f'<p class="tpc-empty">{escape(message)}</p>'

  out = []
  for item in items:
    name = item.get("name", "N/A")
    address = item.get("address", "")
    city = str(item.get("search_city") or item.get("city") or "").strip()
    state = str(item.get("state") or "").strip()
    phone = item.get("phone", "")
    hours = item.get("opening_hours", "")
    website = item.get("website", "")
    meta = ""
    if city and city != "N/A":
      city_label = city
      if state and state != "N/A" and state.lower() != city.lower():
        city_label = f"{city}, {state}"
      meta += f'<span class="tpc-badge">{ICON_CITY} {escape(city_label)}</span>'
    if phone:
      meta += f'<span class="tpc-badge">{ICON_PHONE} {phone}</span>'
    if hours:
      meta += f'<span class="tpc-badge">{ICON_CLOCK} {hours}</span>'
    if website:
      meta += f'<a class="tpc-badge" href="{website}" target="_blank">{ICON_GLOBE} Website</a>'
    meta_html = f'<div class="tpc-card-meta">{meta}</div>' if meta else ""
    out.append(
      f'<div class="tpc-card">'
      f'<p class="tpc-card-name">{icon} {name}</p>'
      f'<p class="tpc-card-addr">{ICON_PIN} {address}</p>'
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


def _image_dedupe_key(image_url: str) -> str:
  if "/wikipedia/commons/thumb/" in image_url:
    prefix, tail = image_url.split("/wikipedia/commons/thumb/", 1)
    parts = tail.split("/")
    if len(parts) >= 4:
      return prefix + "/wikipedia/commons/" + "/".join(parts[:3])
  return image_url


def _wikipedia_image_urls(term: str, verify_ssl: bool | str, limit: int = 5) -> list[str]:
  if not term:
    return []

  requests_to_try = [
    {
      "action": "query",
      "titles": term,
      "prop": "pageimages",
      "piprop": "original|thumbnail",
      "pithumbsize": 1400,
      "format": "json",
    },
  ]

  for search_term in (term, f"{term} landmarks", f"{term} tourism"):
    requests_to_try.append(
      {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrlimit": min(limit, 5),
        "prop": "pageimages",
        "piprop": "original|thumbnail",
        "pithumbsize": 1400,
        "format": "json",
      }
    )

  image_urls: list[str] = []
  seen: set[str] = set()

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
      for candidate in (
        page.get("original", {}).get("source"),
        page.get("thumbnail", {}).get("source"),
      ):
        candidate_key = _image_dedupe_key(candidate or "")
        if candidate and candidate_key not in seen:
          seen.add(candidate_key)
          image_urls.append(candidate)
          if len(image_urls) >= limit:
            return image_urls

  return image_urls


def _location_photo_urls(place: dict, limit: int = 5) -> list[str]:
  verify_ssl = getattr(pf, "_SSL_VERIFY", True)
  search_terms: list[str] = []
  fallback_name = str(place.get("name", "")).strip()
  has_location_qualifier = any(
    str(place.get(key, "")).strip() and str(place.get(key, "")).strip() != "N/A"
    for key in ("city", "state", "country")
  )
  candidate_terms = [
    _joined_place_terms(place.get("name"), place.get("city"), place.get("country")),
    _joined_place_terms(place.get("city"), place.get("state"), place.get("country")),
    _joined_place_terms(place.get("name"), place.get("country")),
    str(place.get("formatted_address", "")).strip(),
  ]

  if fallback_name and not has_location_qualifier:
    candidate_terms.append(fallback_name)

  for term in candidate_terms:
    if term and term not in search_terms:
      search_terms.append(term)

  image_urls: list[str] = []
  seen: set[str] = set()
  for term in search_terms:
    for image_url in _wikipedia_image_urls(term, verify_ssl, limit=limit):
      image_key = _image_dedupe_key(image_url)
      if image_url and image_key not in seen:
        seen.add(image_key)
        image_urls.append(image_url)
        if len(image_urls) >= limit:
          return image_urls

  return image_urls


def _location_photo_url(place: dict) -> str:
  image_urls = _location_photo_urls(place, limit=1)
  return image_urls[0] if image_urls else ""


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
    image_url: str = "",
    image_urls: list[str] | None = None,
    alt_text: str,
    caption: str,
    empty_text: str,
    action_url: str = "",
    action_label: str = "",
) -> str:
    media_urls = [url for url in (image_urls or ([] if not image_url else [image_url])) if url]

    if not media_urls:
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
        f'{ICON_LINK} {escape(action_label)}</a>'
            '</div>'
        )

    media_html = ""
    if len(media_urls) == 1:
        media_html = (
            f'<img class="tpc-location-media" src="{escape(media_urls[0], quote=True)}" '
            f'alt="{escape(alt_text, quote=True)}" loading="lazy">'
        )
    else:
        slider_id = "tpc-loc-" + hashlib.md5("|".join(media_urls).encode("utf-8")).hexdigest()[:10]
        count = len(media_urls)
        inputs_html: list[str] = []
        controls_html: list[str] = []
        dots_html: list[str] = []
        rules: list[str] = []
        slides_html: list[str] = []
        for idx, media_url in enumerate(media_urls):
            checked_attr = " checked" if idx == 0 else ""
            input_id = f"{slider_id}-{idx}"
            prev_idx = (idx - 1) % count
            next_idx = (idx + 1) % count
            inputs_html.append(
                f'<input class="tpc-location-slide-input" type="radio" name="{slider_id}" id="{input_id}"{checked_attr}>'
            )
            slides_html.append(
                '<div class="tpc-location-slide">'
                f'<img class="tpc-location-media" src="{escape(media_url, quote=True)}" '
                f'alt="{escape(alt_text, quote=True)} {idx + 1}" loading="lazy">'
                '</div>'
            )
            controls_html.append(
                f'<label class="tpc-location-slider-btn prev" data-slide="{idx}" for="{slider_id}-{prev_idx}" aria-label="Previous image">&#8249;</label>'
            )
            controls_html.append(
                f'<label class="tpc-location-slider-btn next" data-slide="{idx}" for="{slider_id}-{next_idx}" aria-label="Next image">&#8250;</label>'
            )
            dots_html.append(
                f'<label class="tpc-location-slider-dot" for="{input_id}" aria-label="Show image {idx + 1}"></label>'
            )
            rules.append(
              f'#{input_id}:checked ~ .tpc-location-slider-shell .tpc-location-slider-track ' + '{ transform: translateX(-' + f'{idx * 100}%' + '); }'
            )
            rules.append(
                f'#{input_id}:checked ~ .tpc-location-slider-shell .tpc-location-slider-controls label[data-slide="{idx}"] ' + '{ opacity: 1; pointer-events: auto; }'
            )
            rules.append(
                f'#{input_id}:checked ~ .tpc-location-slider-shell .tpc-location-slider-dots label[for="{input_id}"] ' + '{ background: #ffffff; }'
            )

        media_html = (
            '<div class="tpc-location-slider">'
            f'<style>{"".join(rules)}</style>'
            f'{"".join(inputs_html)}'
            '<div class="tpc-location-slider-shell">'
            f'<div class="tpc-location-slider-track">{"".join(slides_html)}</div>'
            f'<div class="tpc-location-slider-controls">{"".join(controls_html)}</div>'
            f'<div class="tpc-location-slider-dots">{"".join(dots_html)}</div>'
            '</div>'
            '</div>'
        )

    return (
        '<div class="tpc-location-panel">'
        f'{media_html}'
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
      f'{ICON_PIN} <strong>{escape(str(place.get("name", "N/A")))}</strong>'
      f'&ensp;&middot;&ensp;{escape(str(place.get("city", "N/A")))}, '
        f'{escape(str(place.get("state", "N/A")))}, {escape(str(place.get("country", "N/A")))}<br>'
        f'<small style="opacity:.85">{escape(str(place.get("formatted_address", "N/A")))}'
      f'&ensp;&middot;&ensp;{escape(str(place.get("place_type", "N/A")))}</small>'
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
    err = f'<p style="color:#ef4444;padding:12px">{ICON_WARNING} {exc}</p>'

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
      err,
    )

  p = data["place"]
  name = str(p.get("name", "this location"))
  photo_urls = _location_photo_urls(p, limit=3)
  map_url = _location_map_url(p)
  map_link = ""
  if p.get("lat") is not None and p.get("lon") is not None:
    map_link = (
      "https://www.google.com/maps/search/?api=1&query="
      f'{p["lat"]},{p["lon"]}'
    )

  location_image_html = _location_panel_html(
    image_urls=photo_urls,
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

  tab_meta = {
      "attractions": (ICON_ATTRACTIONS, None),
      "museums": (ICON_MUSEUM, None),
      "restaurants": (ICON_RESTAURANT, None),
      "hotels": (ICON_HOTEL, None),
      "parks": (ICON_PARK, None),
      "cities": (ICON_CITY, "No cities found for this region."),
  }
  tabs = tuple(
      _cards(data.get(key, []), tab_meta[key][0], tab_meta[key][1])
      for _, key in _TABS
  )
  has_nearby_results = any(bool(data.get(key)) for _, key in _TABS)
  return (
    gr.update(visible=True),
    gr.update(visible=has_nearby_results),
    location_image_html,
    location_details_html,
    location_map_html,
  ) + tabs


# â”€â”€ event handlers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
  "",
)


def on_text_search(place: str):
  resolved_place = _resolve_place_query(place)
  if not resolved_place:
        return BLANK
  return _explore(resolved_place)


def on_image_search(image_source: str):
  if not image_source or not str(image_source).strip():
    return ("",) + BLANK
  detected = ir.recognize_place_from_image(str(image_source).strip())
  return (detected,) + _explore(detected)


# â”€â”€ UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_TABS = [
  (f"{ICON_ATTRACTIONS} Attractions",  "attractions"),
  (f"{ICON_MUSEUM} Museums",      "museums"),
  (f"{ICON_RESTAURANT} Restaurants",  "restaurants"),
  (f"{ICON_HOTEL} Hotels",       "hotels"),
  (f"{ICON_PARK} Parks",        "parks"),
  (f"{ICON_CITY} Cities",       "cities"),
]

with gr.Blocks(css=CSS, js=CAROUSEL_JS, theme=gr.themes.Soft(), title="Travel Planner") as demo:

    # â”€â”€ carousel â”€â”€
  gr.HTML(
    CAROUSEL_HTML,
    elem_id="tpc-carousel-block",
    elem_classes=["tpc-component", "tpc-carousel-block"],
  )

  with gr.Column(elem_classes=["tpc-component", "tpc-main"]):
    with gr.Column(elem_classes=["tpc-component", "tpc-content"]):

      # â”€â”€ text search: input above button (column) â”€â”€
      with gr.Column(
        elem_id="tpc-search-col",
        elem_classes=["tpc-component", "tpc-search-col"],
      ):
        gr.HTML(
          RECENT_SEARCHES_HTML,
          elem_id="tpc-recent-searches",
          elem_classes=["tpc-component", "tpc-recent-searches"],
        )
        txt = gr.Textbox(
          placeholder=f"{ICON_SEARCH}  Paris, Mumbai, Taj Mahal, Goa...",
          label="Search by Place Name",
          show_label=False,
          container=False,
          elem_id="tpc-search-input",
          elem_classes=["tpc-component", "tpc-search-input"],
        )
        search_btn = gr.Button(
          f"Explore {ICON_PLANE}", variant="primary",
          elem_id="tpc-search-btn",
          elem_classes=["tpc-component", "tpc-search-btn"],
        )

      # â”€â”€ image upload â”€â”€
      with gr.Accordion(
        f"{ICON_CAMERA}  Detect place from a photo",
        open=False,
        elem_id="tpc-upload-acc",
        elem_classes=["tpc-component", "tpc-upload-acc"],
      ):
        with gr.Row(elem_classes=["tpc-component", "tpc-upload-row"]):
          img_data = gr.Textbox(
            value="",
            visible=True,
            show_label=False,
            elem_id="tpc-image-data",
            elem_classes=["tpc-component", "tpc-image-data"],
          )
          gr.HTML(
            UPLOAD_HTML,
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

      # â”€â”€ current location tabs â”€â”€
      with gr.Column(
        visible=False,
        elem_classes=["tpc-component", "tpc-location-section"],
      ) as location_section:
        with gr.Tabs(
          elem_id="tpc-location-tabs",
          elem_classes=["tpc-component", "tpc-location-tabs"],
        ):
          with gr.Tab(f"{ICON_IMAGE} Image", elem_classes=["tpc-component", "tpc-location-tab"]):
            location_image_out = gr.HTML(
              elem_classes=["tpc-component", "tpc-location-output"]
            )
          with gr.Tab(f"{ICON_PIN} Details", elem_classes=["tpc-component", "tpc-location-tab"]):
            location_details_out = gr.HTML(
              elem_classes=["tpc-component", "tpc-location-output"]
            )
          with gr.Tab(f"{ICON_MAP} Map", elem_classes=["tpc-component", "tpc-location-tab"]):
            location_map_out = gr.HTML(
              elem_classes=["tpc-component", "tpc-location-output"]
            )

      # â”€â”€ tabs â”€â”€
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

    # â”€â”€ wire events â”€â”€
    text_outs = [
      location_section,
      results_section,
      location_image_out,
      location_details_out,
      location_map_out,
    ] + tab_outputs

    search_btn.click(fn=on_text_search, inputs=txt, outputs=text_outs, queue=False)
    txt.submit(fn=on_text_search, inputs=txt, outputs=text_outs, queue=False)

    img_btn.click(
        fn=on_image_search,
      inputs=img_data,
        outputs=[detected] + text_outs,
      queue=False,
    )

if __name__ == "__main__":
  demo.launch(
    server_port=7860,
    share=False,
    show_error=True,
    allowed_paths=[str(BASE)],
  )
