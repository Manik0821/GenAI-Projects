"""Geoapify Places API fetcher for Travel Planner."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlencode

import requests
from requests.structures import CaseInsensitiveDict
import urllib3
from dotenv import load_dotenv

# Load workspace-level .env
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

GEOAPIFY_BASE_URL = "https://api.geoapify.com/v2/places"
GEOAPIF_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"

_HEADERS = CaseInsensitiveDict({"Accept": "application/json"})

# Set REQUESTS_VERIFY_SSL=false in .env to disable SSL verification (e.g. corporate proxy)
_SSL_VERIFY: bool | str = (
    False
    if os.getenv("REQUESTS_VERIFY_SSL", "true").strip().lower() == "false"
    else True
)

if not _SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _get_api_key() -> str:
    key = os.getenv("GEOAPIFY_API_KEY", "")
    if not key:
        raise ValueError("GEOAPIFY_API_KEY not found in .env")
    return key


def fetch_places(
    *,
    categories: str | list[str],
    bbox: tuple[float, float, float, float],
    limit: int = 20,
    conditions: str | None = None,
) -> dict:
    """Fetch places from Geoapify Places API.

    Args:
        categories: One or more Geoapify category strings,
                    e.g. "catering.restaurant" or ["tourism.attraction", "leisure.park"].
        bbox: (lon_min, lat_min, lon_max, lat_max) bounding box.
        limit: Maximum number of results (1-500).
        conditions: Optional extra filter string, e.g. "named".

    Returns:
        Parsed JSON response dict with a "features" list.
    """
    if isinstance(categories, list):
        category_str = ",".join(categories)
    else:
        category_str = categories

    lon_min, lat_min, lon_max, lat_max = bbox
    filter_value = f"rect:{lon_min},{lat_min},{lon_max},{lat_max}"

    params: dict[str, str | int] = {
        "categories": category_str,
        "filter": filter_value,
        "limit": limit,
        "apiKey": _get_api_key(),
    }
    if conditions:
        params["conditions"] = conditions

    resp = requests.get(GEOAPIFY_BASE_URL, headers=_HEADERS, params=params, timeout=15, verify=_SSL_VERIFY)
    resp.raise_for_status()
    return resp.json()


def fetch_places_by_radius(
    *,
    categories: str | list[str],
    lat: float,
    lon: float,
    radius_meters: int = 5000,
    limit: int = 20,
    conditions: str | None = None,
) -> dict:
    """Fetch places within a circle around a point.

    Args:
        categories: One or more Geoapify category strings.
        lat: Center latitude.
        lon: Center longitude.
        radius_meters: Search radius in metres.
        limit: Maximum number of results.
        conditions: Optional extra filter string.

    Returns:
        Parsed JSON response dict with a "features" list.
    """
    if isinstance(categories, list):
        category_str = ",".join(categories)
    else:
        category_str = categories

    filter_value = f"circle:{lon},{lat},{radius_meters}"

    params: dict[str, str | int] = {
        "categories": category_str,
        "filter": filter_value,
        "limit": limit,
        "apiKey": _get_api_key(),
    }
    if conditions:
        params["conditions"] = conditions

    resp = requests.get(GEOAPIFY_BASE_URL, headers=_HEADERS, params=params, timeout=15, verify=_SSL_VERIFY)
    resp.raise_for_status()
    return resp.json()


def geocode_place_name(place_name: str) -> tuple[float, float]:
    """Convert a place name to (lat, lon) using Geoapify Geocoding API.

    Args:
        place_name: Any free-text location, e.g. "Paris", "Goa, India", "Eiffel Tower".

    Returns:
        (lat, lon) of the best match.

    Raises:
        ValueError: if no results are returned for the given name.
    """
    params = {
        "text": place_name,
        "limit": 1,
        "apiKey": _get_api_key(),
    }
    resp = requests.get(
        GEOAPIF_GEOCODE_URL, headers=_HEADERS, params=params, timeout=15, verify=_SSL_VERIFY
    )
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])
    if not features:
        raise ValueError(f"No geocoding results found for '{place_name}'")
    coords = features[0]["geometry"]["coordinates"]  # [lon, lat]
    return float(coords[1]), float(coords[0])  # return as (lat, lon)


def fetch_places_by_name(
    place_name: str,
    *,
    categories: str | list[str],
    radius_meters: int = 5000,
    limit: int = 20,
    conditions: str | None = None,
) -> dict:
    """Fetch places near a named location.

    Geocodes the place name to coordinates first, then searches for nearby
    places matching the given categories within the specified radius.

    Args:
        place_name: Human-readable location, e.g. "Goa", "Bangalore", "Times Square".
        categories: Geoapify category string(s), e.g. "catering.restaurant".
        radius_meters: Search radius around the resolved point (default 5 km).
        limit: Maximum number of results.
        conditions: Optional extra filter, e.g. "named".

    Returns:
        Parsed JSON response dict with a "features" list, plus a "_resolved" key
        containing the geocoded (lat, lon) used for the search.
    """
    lat, lon = geocode_place_name(place_name)
    result = fetch_places_by_radius(
        categories=categories,
        lat=lat,
        lon=lon,
        radius_meters=radius_meters,
        limit=limit,
        conditions=conditions,
    )
    result["_resolved"] = {"place_name": place_name, "lat": lat, "lon": lon}
    return result


def geocode_place_details(place_name: str) -> dict:
    """Return rich metadata about a place from the Geoapify Geocoding API.

    Returns a dict with keys: name, formatted_address, city, state, county,
    country, country_code, postcode, lat, lon, place_id, place_type.
    """
    params = {
        "text": place_name,
        "limit": 1,
        "apiKey": _get_api_key(),
    }
    resp = requests.get(
        GEOAPIF_GEOCODE_URL, headers=_HEADERS, params=params, timeout=15, verify=_SSL_VERIFY
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError(f"No geocoding results found for '{place_name}'")
    props = features[0].get("properties", {})
    coords = features[0]["geometry"]["coordinates"]
    return {
        "name": props.get("name") or props.get("city") or props.get("county") or place_name,
        "formatted_address": props.get("formatted", "N/A"),
        "city": props.get("city", "N/A"),
        "state": props.get("state", "N/A"),
        "county": props.get("county", "N/A"),
        "country": props.get("country", "N/A"),
        "country_code": props.get("country_code", "N/A"),
        "postcode": props.get("postcode", "N/A"),
        "lat": float(coords[1]),
        "lon": float(coords[0]),
        "place_id": props.get("place_id", ""),
        "place_type": props.get("result_type", "N/A"),
    }


# Categories fetched by explore_place, in display order
_EXPLORE_CATEGORIES: list[tuple[str, str]] = [
    ("attractions",  "tourism.attraction,tourism.sights"),
    ("museums",      "entertainment.museum"),
    ("restaurants",  "catering.restaurant"),
    ("hotels",       "accommodation.hotel"),
    ("parks",        "leisure.park,natural"),
]


def explore_place(
    place_name: str,
    *,
    radius_meters: int = 5000,
    limit_per_category: int = 5,
) -> dict:
    """Fetch comprehensive details about a place and famous spots near it.

    Makes one geocoding call for place metadata and one Places API call per
    category (attractions, museums, restaurants, hotels, parks).

    Args:
        place_name: Human-readable name, e.g. "Paris", "Delhi", "Taj Mahal, Agra".
        radius_meters: Search radius around the resolved centre (default 5 km).
        limit_per_category: Max results returned per category.

    Returns:
        Dict with keys:
            ``place``       — rich metadata about the location itself.
            ``attractions`` — tourist attractions & sights.
            ``museums``     — museums.
            ``restaurants`` — dining options.
            ``hotels``      — accommodation.
            ``parks``       — parks and natural spots.
            ``summary``     — one-line text summary of counts.
    """
    place_info = geocode_place_details(place_name)
    lat, lon = place_info["lat"], place_info["lon"]

    result: dict = {"place": place_info}
    counts: list[str] = []

    for key, category_str in _EXPLORE_CATEGORIES:
        try:
            raw = fetch_places_by_radius(
                categories=category_str,
                lat=lat,
                lon=lon,
                radius_meters=radius_meters,
                limit=limit_per_category,
                conditions="named",
            )
            items = [extract_place_details(f) for f in raw.get("features", [])]
        except Exception:
            items = []
        result[key] = items
        counts.append(f"{len(items)} {key}")

    result["summary"] = (
        f"{place_info['name']} ({place_info['country']}) | " + ", ".join(counts)
    )
    return result


def extract_place_details(feature: dict) -> dict:
    """Pull the most useful fields out of a single GeoJSON feature."""
    props = feature.get("properties", {})
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates", [None, None])
    return {
        "name": props.get("name", "N/A"),
        "categories": props.get("categories", []),
        "address": props.get("formatted", "N/A"),
        "city": props.get("city", "N/A"),
        "country": props.get("country", "N/A"),
        "postcode": props.get("postcode", "N/A"),
        "lon": coordinates[0],
        "lat": coordinates[1],
        "place_id": props.get("place_id", ""),
        "website": props.get("website", ""),
        "phone": props.get("phone", ""),
        "opening_hours": props.get("opening_hours", ""),
    }


if __name__ == "__main__":
    import json
    data = explore_place("Panaji, Goa", radius_meters=5000, limit_per_category=3)
    print("Summary:", data["summary"])
    print("\nPlace info:")
    for k, v in data["place"].items():
        print(f"  {k}: {v}")
    for section in ("attractions", "museums", "restaurants", "hotels", "parks"):
        items = data[section]
        print(f"\n{section.upper()} ({len(items)}):")
        for item in items:
            print(f"  - {item['name']} | {item['address']}")
