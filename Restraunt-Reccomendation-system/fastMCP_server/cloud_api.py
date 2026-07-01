from fastapi import FastAPI
from pydantic import BaseModel

from server import (
    recommend_by_vibe,
    get_restaurant_info,
    get_review,
)

app = FastAPI(title="Connoisseur Companion API")


class RecommendRequest(BaseModel):
    vibe: str


class RestaurantRequest(BaseModel):
    restaurant_name: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "connoisseur-companion-api"}


@app.post("/recommend_by_vibe")
def api_recommend_by_vibe(payload: RecommendRequest):
    return {"result": recommend_by_vibe(payload.vibe)}


@app.post("/get_restaurant_info")
def api_get_restaurant_info(payload: RestaurantRequest):
    return {"result": get_restaurant_info(payload.restaurant_name)}


@app.post("/get_review")
def api_get_review(payload: RestaurantRequest):
    return {"result": get_review(payload.restaurant_name)}