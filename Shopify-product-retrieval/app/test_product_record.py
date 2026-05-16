import json
from app.ingest import ingest_product_url
from app.evaluator import (
    handle_reviews_check,
    handle_ingredient_lookup,
    handle_page_improvement_audit
)

# 1. Fetch live product data from Healf
# product_url = "https://healf.com"
product_url = "https://healf.com/en-uk/products/creatine?variant=43722624467183&selling_plan=6501073135"

print(f"📥 Ingesting product data from: {product_url}...\n")
product_record = ingest_product_url(product_url)


# 2. Test Handler 1: Reviews Check
print("=========================================")
print("🧪 TESTING: handle_reviews_check")
print("=========================================")
reviews_output = handle_reviews_check(product_record)
print(reviews_output)
print("\n")


# 3. Test Handler 2: Ingredient Lookup
print("=========================================")
print("🧪 TESTING: handle_ingredient_lookup")
print("=========================================")
# Looking for B12 based on the new product URL
test_ingredient = "b12" 
ingredient_output = handle_ingredient_lookup(product_record, test_ingredient)
print(ingredient_output)
print("\n")


# 4. Test Handler 3: Page Improvement Audit
print("=========================================")
print("🧪 TESTING: handle_page_improvement_audit")
print("=========================================")
audit_output = handle_page_improvement_audit(product_record)
print(audit_output)
print("=========================================")
