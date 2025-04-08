from fastapi import APIRouter, Query
from typing import List
from app.recommender import recommend_basic, recommend_selected

router = APIRouter()

@router.get("/recommend/basic")
def recommend_basic_api(
    user_ott: List[str] = Query(...),
    user_genre: List[str] = Query(...),
    total_needed: int = 5,
    prefer_new: bool = False
):
    return {"results": recommend_basic(user_ott, user_genre, total_needed, prefer_new)}

@router.get("/recommend/selected")
def recommend_selected_api(
    selected_title: str,
    total_needed: int = 5
):
    return {"results": recommend_selected(selected_title, total_needed)}
