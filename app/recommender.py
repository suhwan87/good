import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from collections import defaultdict

# 데이터 로딩 및 전처리
df = pd.read_csv("data/CONTENTS_FIN.csv")
df['CONTENTS_GENRE'] = df['CONTENTS_GENRE'].fillna('').apply(lambda x: x.split(', '))
df['OTT'] = df['OTT'].fillna('').apply(lambda x: x.split(', '))
df['DIRECTOR'] = df['DIRECTOR'].fillna('')
df['CAST'] = df['CAST'].fillna('')
df['RELEASE_YEAR'] = df['RELEASE_YEAR'].fillna('0').astype(str).str.extract(r'(\d{4})').fillna(0).astype(int)

# 벡터 준비
mlb_ott = MultiLabelBinarizer()
mlb_genre = MultiLabelBinarizer()
ott_vec = mlb_ott.fit_transform(df['OTT'])
genre_vec = mlb_genre.fit_transform(df['CONTENTS_GENRE'])
scaler = MinMaxScaler()
year_vec = scaler.fit_transform(df[['RELEASE_YEAR']])
basic_content_vec = np.hstack([ott_vec, genre_vec, year_vec])

vectorizer_director = CountVectorizer()
vectorizer_cast = CountVectorizer()
genre_vec_sel = mlb_genre.transform(df['CONTENTS_GENRE'])
director_vec = vectorizer_director.fit_transform(df['DIRECTOR'])
cast_vec = vectorizer_cast.fit_transform(df['CAST'])
selected_content_vec = hstack([genre_vec_sel, director_vec, cast_vec]).tocsr()


def recommend_basic(user_ott, user_genre, prefer_new=True, total_needed=5):
    user_ott_vec = mlb_ott.transform([user_ott])
    user_genre_vec = mlb_genre.transform([user_genre])
    user_year_vec = [[1.0 if prefer_new else 0.0]]
    user_vec = np.hstack([user_ott_vec, user_genre_vec, user_year_vec])
    
    sims = cosine_similarity(user_vec, basic_content_vec)[0]
    df['유사도'] = sims

    # ✅ 유사도 상위 50개만 선별한 뒤 셔플
    top_df = df.sort_values(by='유사도', ascending=False).head(50).sample(frac=1).copy()

    genres_selected = len(user_genre)
    min_per_genre = total_needed // genres_selected
    extra = total_needed % genres_selected
    genre_limit = {g: min_per_genre + (1 if i < extra else 0) for i, g in enumerate(user_genre)}
    
    recommendations = []
    genre_count = defaultdict(int)

    for idx, row in top_df.iterrows():
        content_genres = row['CONTENTS_GENRE']
        matched = [g for g in content_genres if g in user_genre]
        
        for g in matched:
            if genre_count[g] < genre_limit[g]:
                recommendations.append(row)
                genre_count[g] += 1
                break
        
        if len(recommendations) >= total_needed:
            break

    # ✅ 부족하면 top_df에서 남은 것 채우기
    if len(recommendations) < total_needed:
        already_selected = set(r.name for r in recommendations)
        remaining = top_df[~top_df.index.isin(already_selected)]
        if prefer_new:
            remaining = remaining.sort_values(by='RELEASE_YEAR', ascending=False)
        more = remaining.head(total_needed - len(recommendations)).to_dict(orient='records')
        recommendations.extend(more)

    result_df = pd.DataFrame(recommendations)
    return result_df[['CONTENTS_TITLE', 'CONTENTS_GENRE', 'OTT', 'RELEASE_YEAR', '유사도']].to_dict(orient='records')



def recommend_selected(title: str, top_n: int = 5):
    if title not in df['CONTENTS_TITLE'].values:
        return {"message": f"'{title}'은(는) 데이터에 없습니다."}
    
    idx = df[df['CONTENTS_TITLE'] == title].index[0]
    sim_scores = cosine_similarity(selected_content_vec[idx], selected_content_vec).flatten()
    
    df['유사도'] = sim_scores
    result = df[df.index != idx].sort_values(by='유사도', ascending=False).head(top_n)
    
    return result[['CONTENTS_TITLE', 'CONTENTS_GENRE', 'DIRECTOR', 'CAST', '유사도']].to_dict(orient='records')
