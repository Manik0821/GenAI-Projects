# Loading-Data - Data Ingestion

Scripts and utilities for loading raw restaurant and recipe data from various sources.

## Files

- **loading_data.py** - Main data loader
  - Reads raw restaurant descriptions from California-Culinary-Map.txt
  - Parses structure: Restaurant name, location, cuisines, vibes, ratings
  - Converts to structured JSON format

- **import_libs.py** - Library discovery and validation
  - Checks required dependencies (pandas, numpy, langchain, etc.)
  - Validates API access (NVIDIA, OpenAI)
  - Reports missing packages

- **California-Culinary-Map.txt** - Source data
  - 100+ California restaurants with descriptions
  - Includes cuisine types, neighborhoods, ratings, price ranges
  - Raw text format for preprocessing

## Output

Generates `structured_restaurant_data.json`:
```json
{
  "name": "Restaurant Name",
  "neighborhood": "Santa Monica",
  "location": "Santa Monica",
  "cuisine": "Modern Italian",
  "vibes": ["cozy", "candlelit", "trattoria"],
  "rating": 4.5,
  "price_range": 3,
  "description": "Full description..."
}
```

## Usage

```bash
python loading_data.py
# Reads California-Culinary-Map.txt
# Outputs structured_restaurant_data.json
```

## Notes

- Data is automatically enriched in the pipeline
- Cuisine inference happens in server.py if missing
- Ratings and price ranges parsed from descriptions
- Output drives both recommend_by_vibe and vector DB indexing
