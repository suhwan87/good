from fastapi import APIRouter, Query
from typing import List
from app.recommend import hybrid_recommendation

router = APIRouter()

@router.get("/recommend")
def recommend(
    user_ott: List[str] = Query(...),
    user_genre: List[str] = Query(...),
    selected_title: str = "",
    total_needed: int = 5,
    prefer_new: bool = False
):
    results = hybrid_recommendation(
        user_ott=user_ott,
        user_genre=user_genre,
        selected_title=selected_title,
        total_needed=total_needed,
        prefer_new=prefer_new
    )
    return {"results": results}
