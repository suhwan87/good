from fastapi import APIRouter, Query
from typing import List, Optional
from app.recommender import hybrid_recommendation

router = APIRouter()

@router.get("/")
def root():
    return {"message": "OTT Recommendation API"}

@router.get("/recommend")
def get_recommendation(
    user_ott: List[str] = Query(...),
    user_genre: List[str] = Query(...),
    selected_title: Optional[str] = None,
    prefer_new: bool = False,
    total_needed: int = 5
):
    results = hybrid_recommendation(
        user_ott=user_ott,
        user_genre=user_genre,
        selected_title=selected_title,
        total_needed=total_needed,
        prefer_new=prefer_new
    )
    return {"results": results}
