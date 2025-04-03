from fastapi import APIRouter, Query
from typing import List
from app.recommender import OTTRecommender

router = APIRouter()
recommender = OTTRecommender("data/OTT_contents_list.csv")

@router.get("/recommend")
def get_recommendations(
    user_ott: List[str] = Query(...),
    user_genre: List[str] = Query(...),
    total_needed: int = 10
):
    df = recommender.recommend(user_ott, user_genre, total_needed)
    return df.to_dict(orient="records")
