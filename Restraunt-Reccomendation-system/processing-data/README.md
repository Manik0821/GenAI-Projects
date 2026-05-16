# Processing-Data - Data Enrichment

Scripts for processing, enriching, and extracting features from raw restaurant and review data.

## Files

- **process_data.py** - Data processing pipeline
  - Text cleaning and normalization
  - Feature extraction (cuisine, vibes, dietary)
  - Review sentiment analysis
  - Image processing preparation
  - Data validation and quality checks

## Processing Steps

1. **Text Normalization**
   - Remove special characters
   - Standardize whitespace
   - Case normalization

2. **Feature Extraction**
   - Detect cuisine types from descriptions
   - Extract vibe tags
   - Identify dietary accommodations
   - Parse ratings and prices

3. **Enrichment**
   - Add inferred cuisines
   - Expand vibe categories
   - Link reviews to restaurants
   - Validate geographic data

4. **Output**
   - Cleaned structured JSON
   - Ready for vector indexing
   - Ready for restaurant searches

## Usage

```bash
python process_data.py
# Input: raw California-Culinary-Map.txt, reviews
# Output: enriched structured_restaurant_data.json
```

## Quality Checks

- Validates required fields: name, location, cuisine
- Ensures ratings in 0.0-5.0 range
- Verifies price ranges 1-4
- Confirms vibe tags are non-empty

## Notes

- Processing is idempotent (safe to run multiple times)
- Enrichment doesn't overwrite explicit data
- Output feeds into vectorDB indexing and app searches
