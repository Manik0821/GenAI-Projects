# Healf Product Intelligence Agent (MVP)

This project is a functional MVP that answers natural-language questions about live Healf product pages.

## What It Does Today

1. Navigate
- Accepts a live Healf product URL and fetches page HTML.
- Normalizes URL and extracts locale/product handle.

2. Ingest
- Extracts multiple facets from live pages:
  - Product text (title, headings, sections, page text)
  - Reviews signals (JSON-LD AggregateRating + page review indicators)
  - Shopify structured product JSON via .js endpoint

3. Evaluate
- Routes user question into intents such as:
  - product_overview (default for non-specific queries)
  - reviews_check
  - ingredients_list
  - ingredients_with_alternatives
  - ingredient_lookup
  - page_improvement_audit
- Uses deterministic evaluators plus LLM reasoning for page audits.
- Returns confidence level and evidence-backed sources for each answer.

4. Act on Findings
- Returns direct answers and recommendations in natural language.
- Can emit JSON output for automation.
- Supports clarification prompts when user intent is incomplete (for example, missing ingredient entity).
- Supports multi-turn chat on the same product URL.

## Architecture

- app/agent.py
  - CLI entrypoint for URL + natural-language question
  - Interactive chat mode with follow-up questions and URL switching
- app/ingest.py
  - End-to-end product ingestion pipeline
- app/fetchers.py
  - HTML and Shopify .js fetching
- app/parsers.py
  - JSON-LD/text/review/ingredients extraction
- app/router.py
  - Intent routing from plain-English question
  - Specific-query routing vs non-specific fallback to product overview
- app/evaluator.py
  - Intent handlers + LLM-backed audit synthesis
  - Product overview generation (details, aggregate reviews, strengths, weaknesses, extra details)
  - Ingredients listing and ingredient-similarity alternatives
  - Structured response metadata: confidence, sources, evidence, clarification
- app/models.py
  - NVIDIA-hosted LLM client setup

## Run

From project root:

```powershell
$env:PYTHONPATH="."
python app/agent.py --url "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791" --question "Does this product have any reviews?"

python app/agent.py --url "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791" --question "Does this product have Vitamin D in it?"

python app/agent.py --url "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791" --question "What can I improve on this page?"

python app/agent.py --url "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791" --question "show me all the ingredients"

python app/agent.py --url "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791" --question "suggest alternatives with similar ingredients" --json

python app/agent.py --url "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791" --question "tell me about this product" --json

# Multi-turn conversational mode
python app/agent.py --chat --url "https://healf.com/en-uk/products/lmnt-recharge-electrolytes-variety-pack?variant=44024590893295&selling_plan=6835863791"
```

JSON mode:

```powershell
python app/agent.py --url "<healf-pdp-url>" --question "What can I improve on this page?" --json
```

In chat mode:
- `/url <new-product-url>` switches product context
- `/quit` exits the session

## Requirements

- Python 3.10+
- Dependencies in requirements.txt
- NVIDIA API key in .env:
  - NVIDIA_API_KEY (or NVIDIA_API_KEY1/2/3)

## Example Output Types

- Product overview (default fallback):
  - Product details, aggregate reviews, strengths/weaknesses, and body-derived details
- Review check:
  - "Reviews found with aggregate rating X/Y based on N reviews"
- Ingredients list:
  - Parsed ingredients list from extracted ingredient sections/body text
- Ingredient alternatives:
  - Similar-product suggestions by ingredient overlap from sampled live Healf products
- Ingredient lookup:
  - "Found in ingredients section", "Found in page text", or "Not found"
- Page audit:
  - Deterministic strengths/gaps + LLM reasoning summary with prioritized improvements
- Every response includes:
  - confidence (`high`, `medium`, `low`)
  - sources (for example `json_ld`, `visible_html`, `shopify_product_json`, `nvidia_llm`)
  - evidence bullets showing exactly what was used

## Deliverables

- Working MVP code: this repository
- Example outputs from live runs: see `example_outputs.md`
- Architecture + extension plan: this README

## Scope and Improvement Plan

What this MVP already handles:
- Live data ingestion from healf.com
- Conversational question handling (URL + plain English)
- Multi-capability evaluation with practical answers

Known current limitations:
1. Negative-review body extraction is not fully implemented; review answers are currently aggregate-focused.
2. Alternatives are based on sampled sitemap candidates and ingredient overlap, so some relevant alternatives may be missed.
3. LLM reasoning can occasionally fall back to deterministic output when provider connectivity is unavailable.

What to add next (near-term):
1. Stronger review extraction from third-party widgets and dynamic payloads
2. Image quality and information-density scoring for PDP media
3. Better ingredient entity matching (synonyms, forms, dosages)
4. Batch mode and API endpoint for team workflows

Execution plan (next 2 weeks):
1. Add robust review extractor for Judge.me/Yotpo/Loox payloads
  - KPI: improve review detection recall from baseline to >90% on a 100-PDP sample
2. Add explicit source citation blocks in all human-mode answers
  - KPI: 100% of responses include at least one source and one evidence item
3. Add conversational memory for follow-up references ("what about ingredients?")
  - KPI: >85% correct follow-up resolution in scripted chat tests
4. Add regression test set for top question intents
  - KPI: keep intent routing accuracy above 95% for known prompts

Production-ready plan (3 months):
1. Reliability and scale
  - Add caching, retries, circuit breakers, and structured logs/traces
  - SLO target: p95 response < 4s, success rate > 99%
2. Data layer
  - Store normalized product records and diffs for historical comparisons
  - KPI: daily snapshot coverage for top 2,000 PDPs
3. Quality and trust
  - Add strict guardrails and mandatory citations in every answer
  - KPI: hallucination rate < 2% in manual weekly QA spot checks
4. Capability expansion
  - Add pricing/subscription analysis, media quality scoring, and competitor benchmarking
  - KPI: improve accepted recommendation quality score by internal reviewers to >4.3/5
