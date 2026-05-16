from app.router import route_question

examples = [
    "Does this product have any reviews?",
    "Does this product have Vitamin D in it?",
    "What can I improve on this page?",
    "How many reviews does this product have?",
    "Is there magnesium in this product?",
    "Can you audit this page for SEO and content quality?",
]

if __name__ == "__main__":
    for q in examples:
        result = route_question(q)
        print("\nQ:", q)
        print("Route:", result)