from pydantic import BaseModel, Field
from typing import List, Optional, Any


class PriceInfo(BaseModel):
    currency: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    raw_price_text: Optional[str] = None


class ReviewInfo(BaseModel):
    has_reviews: bool = False
    review_count: Optional[int] = None
    average_rating: Optional[float] = None
    evidence: List[str] = Field(default_factory=list)


class ImageInfo(BaseModel):
    url: str
    alt: Optional[str] = None


class ProductRecord(BaseModel):
    source_url: str
    canonical_url: Optional[str] = None
    handle: str
    locale: Optional[str] = None
    variant_id: Optional[str] = None

    title: Optional[str] = None
    brand: Optional[str] = None
    description_text: Optional[str] = None
    headings: List[str] = Field(default_factory=list)

    ingredients_text: List[str] = Field(default_factory=list)
    faq_text: List[str] = Field(default_factory=list)

    price: PriceInfo = Field(default_factory=PriceInfo)
    reviews: ReviewInfo = Field(default_factory=ReviewInfo)
    images: List[ImageInfo] = Field(default_factory=list)

    shopify_json_present: bool = False
    json_ld_present: bool = False

    raw_shopify_json: Optional[Any] = None
    raw_json_ld: List[Any] = Field(default_factory=list)