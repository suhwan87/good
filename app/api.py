from fastapi import APIRouter, Query
from typing import List
from recommender import recommend_basic, recommend_selected

router = APIRouter()


@router.get("/recommend/basic")
def get_basic_recommendation(
    ott: List[str] = Query(...),
    genre: List[str] = Query(...),
    prefer_new: bool = True,
    total_needed: int = 5
):
    return recommend_basic(user_ott=ott, user_genre=genre, prefer_new=prefer_new, total_needed=total_needed)


@router.get("/recommend/selected")
def get_selected_recommendation(
    title: str,
    top_n: int = 5
):
    return recommend_selected(title=title, top_n=top_n)
