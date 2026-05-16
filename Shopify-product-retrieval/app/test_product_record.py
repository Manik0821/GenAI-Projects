from pprint import pprint

from app.ingest import ingest_product_url
from app.normalize import normalize_ingestion_result

URL = "https://healf.com/products/nbpure-methyl-b12-1-oz"

if __name__ == "__main__":
    raw = ingest_product_url(URL)
    record = normalize_ingestion_result(raw)

    pprint(record.model_dump())