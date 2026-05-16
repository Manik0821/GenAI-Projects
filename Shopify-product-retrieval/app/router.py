import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RouteResult:
    intent: str
    ingredient_query: Optional[str] = None


INGREDIENT_LIST_PATTERNS = [
    r"\blist( all)?( the)? ingredients\b",
    r"\blist ingredients\b",
    r"\bshow( me)?( all)?( the)? ingredients\b",
    r"\bwhat are( all)?( the)? ingredients\b",
    r"\bingredients list\b",
]

INGREDIENT_ALTERNATIVE_PATTERNS = [
    r"\bsimilar ingredients\b",
    r"\balternative products\b",
    r"\balternatives?\b",
    r"\bsuggest alternatives?\b",
    r"\bsimilar products\b",
]


REVIEW_PATTERNS = [
    r"\breview\b",
    r"\breviews\b",
    r"\brating\b",
    r"\bratings\b",
    r"\bhow many reviews\b",
    r"\bdoes .* have any reviews\b",
    r"\bdoes .* have reviews\b",
]

OVERVIEW_PATTERNS = [
    r"\babout this product\b",
    r"\bproduct details\b",
    r"\bproduct overview\b",
    r"\btell me about (this|the) product\b",
    r"\bsummarize (this|the) product\b",
    r"\bwhat is this product\b",
]

AUDIT_PATTERNS = [
    r"\bimprove\b",
    r"\baudit\b",
    r"\boptimi[sz]e\b",
    r"\bwhat can i improve\b",
    r"\bhow can this page be better\b",
    r"\bhow can i improve this page\b",
    r"\bwhat is missing\b",
    r"\bquality of (this )?page\b",
    r"\bseo\b",
    r"\bcontent\b",
]

INGREDIENT_PATTERNS = [
    r"\bcontain\b",
    r"\bcontains\b",
    r"\bhave\b",
    r"\bhas\b",
    r"\bingredient\b",
    r"\bingredients\b",
    r"\bvitamin\b",
    r"\bmineral\b",
    r"\bdoes .* have\b",
    r"\bis there\b",
]


def route_question(question: str) -> RouteResult:
    q = normalize_question(question)

    if matches_any(q, OVERVIEW_PATTERNS):
        return RouteResult(intent="product_overview")

    if matches_any(q, INGREDIENT_ALTERNATIVE_PATTERNS):
        return RouteResult(intent="ingredients_with_alternatives")

    if matches_any(q, INGREDIENT_LIST_PATTERNS):
        return RouteResult(intent="ingredients_list")

    if matches_any(q, AUDIT_PATTERNS):
        return RouteResult(intent="page_improvement_audit")

    if looks_like_ingredient_query(q):
        ingredient = extract_ingredient_entity(q)
        return RouteResult(intent="ingredient_lookup", ingredient_query=ingredient)

    if matches_any(q, REVIEW_PATTERNS):
        return RouteResult(intent="reviews_check")

    # Non-specific queries should default to product overview, not page audit.
    return RouteResult(intent="product_overview")


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def looks_like_ingredient_query(q: str) -> bool:
    ingredient_words = [
        "vitamin", "mineral", "magnesium", "zinc", "iron", "b12", "b6", "vitamin d",
        "omega", "creatine", "protein", "electrolyte", "caffeine", "collagen",
        "probiotic", "adaptogen", "ashwagandha", "melatonin", "electrolytes",
    ]

    if any(word in q for word in ingredient_words):
        return True

    if matches_any(q, INGREDIENT_PATTERNS):
        if "review" not in q and "improve" not in q and "audit" not in q:
            return True

    return False

def clean_ingredient_query(text: str) -> str:
    text = text.lower().strip()
    noise_phrases = [
        "in it",
        "in this product",
        "this product",
        "product",
    ]
    for phrase in noise_phrases:
        text = text.replace(phrase, "")
    return re.sub(r"\s+", " ", text).strip(" .,?")


def extract_ingredient_entity(q: str) -> Optional[str]:
    patterns = [
        r"(?:does|do|is|are)\s+.*?\s+(?:have|contain|contains)\s+(.+?)(?:\?|$)",
        r"(?:is there)\s+(.+?)(?:\s+in\s+.*|\?|$)",
        r"(?:does .* have)\s+(.+?)(?:\?|$)",
        r"(?:contains?)\s+(.+?)(?:\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            candidate = match.group(1).strip(" .?,")

            stop_phrases = [
                "reviews", "review", "a review", "any reviews",
                "improvements", "improve", "better", "seo"
            ]
            if candidate and candidate not in stop_phrases:
                return clean_ingredient_query(candidate)

    direct_terms = [
        "vitamin d", "vitamin b12", "b12", "magnesium", "zinc", "iron", "omega 3",
        "omega-3", "creatine", "protein", "caffeine", "collagen", "probiotic",
        "ashwagandha", "melatonin", "electrolytes",
    ]
    for term in direct_terms:
        if term in q:
            return term

    return None