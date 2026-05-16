# Managing-Data - Data Maintenance

Scripts for managing, updating, and enriching structured restaurant metadata.

## Files

- **restaurant_data_management.py** - Data management utilities
  - Add/update restaurant records
  - Backfill missing cuisine information
  - Normalize location names
  - Validate data integrity
  - Merge review data with restaurant records

## Key Functions

- **Add Restaurant**: Insert new restaurant with all fields
- **Update Cuisines**: Infer missing cuisine types from vibes/descriptions
- **Normalize Locations**: Standardize neighborhood names
- **Merge Reviews**: Link customer reviews to restaurants
- **Export Data**: Save cleaned data back to JSON

## Data Schema

```python
{
    "name": str,                  # Restaurant name
    "neighborhood": str,          # Primary location
    "location": str,              # Alternative location field
    "cuisine": str,               # Primary cuisine type
    "food_style": str,            # Alternative cuisine field
    "vibes": List[str],           # Tags: cozy, romantic, casual, upscale...
    "rating": float,              # 0.0-5.0
    "price_range": int,           # 1-4 ($, $$, $$$, $$$$)
    "description": str            # Full text description
}
```

## Usage

```bash
python restaurant_data_management.py
# Runs maintenance tasks on structured_restaurant_data.json
```

## Notes

- Always backup JSON before major updates
- Cuisine inference prioritizes: explicit field → vibe tags → description patterns
- Location normalization is case-insensitive
- Data changes automatically reflect in app (no restart needed)
