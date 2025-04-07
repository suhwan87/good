# app/recommend.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from collections import defaultdict

# 데이터 로딩
df = pd.read_csv("data/CONTENTS_FIN.csv")
df['CONTENTS_GENRE'] = df['CONTENTS_GENRE'].fillna('').apply(lambda x: x.split(', '))
df['OTT'] = df['OTT'].fillna('').apply(lambda x: x.split(', '))
df['DIRECTOR'] = df['DIRECTOR'].fillna('')
df['CAST'] = df['CAST'].fillna('')
df['RELEASE_YEAR'] = df['RELEASE_YEAR'].fillna('0').astype(str).str.extract(r'(\d{4})').fillna(0).astype(int)

# 벡터화
mlb_ott = MultiLabelBinarizer()
mlb_genre = MultiLabelBinarizer()
ott_vec = mlb_ott.fit_transform(df['OTT'])
genre_vec = mlb_genre.fit_transform(df['CONTENTS_GENRE'])
scaler = MinMaxScaler()
year_vec = scaler.fit_transform(df[['RELEASE_YEAR']])
vectorizer_director = CountVectorizer()
director_vec = vectorizer_director.fit_transform(df['DIRECTOR'])
vectorizer_cast = CountVectorizer()
cast_vec = vectorizer_cast.fit_transform(df['CAST'])

# 콘텐츠 벡터
content_vec_initial = np.hstack([ott_vec, genre_vec, year_vec])
content_vec_detailed = hstack([genre_vec, director_vec, cast_vec]).tocsr()

# 추천 함수
def hybrid_recommendation(user_ott, user_genre, selected_title=None, total_needed=5, prefer_new=False):
    user_ott_vec = mlb_ott.transform([user_ott])
    user_genre_vec = mlb_genre.transform([user_genre])
    user_year_vec = [[1.0 if prefer_new else 0.0]]
    user_vec = np.hstack([user_ott_vec, user_genre_vec, user_year_vec])
    sims_init = cosine_similarity(user_vec, content_vec_initial)[0]

    if selected_title and selected_title in df['CONTENTS_TITLE'].values:
        idx = df[df['CONTENTS_TITLE'] == selected_title].index[0]
        sims_selected = cosine_similarity(content_vec_detailed[idx], content_vec_detailed).flatten()
        combined_sims = (sims_init * 0.6) + (sims_selected * 0.4)
        exclude_indices = [idx]
        top_k = 10
    else:
        combined_sims = sims_init
        exclude_indices = []
        top_k = 50

    if prefer_new:
        year_score = scaler.transform(df[['RELEASE_YEAR']]).flatten()
        final_score = (combined_sims * 0.8) + (year_score * 0.2)
    else:
        final_score = combined_sims

    df['유사도'] = final_score
    filtered_df = df[~df.index.isin(exclude_indices)].sort_values(by='유사도', ascending=False).head(top_k)
    sampled_df = filtered_df.sample(n=min(total_needed, len(filtered_df)), replace=False)

    return sampled_df[[
    'CONTENTS_TITLE', 'CONTENTS_GENRE', 'DIRECTOR', 'CAST',
    'OTT', 'RELEASE_YEAR', 'POSTER_IMG', '유사도'
    ]].to_dict(orient='records')

print("학습된 OTT 목록:", mlb_ott.classes_)
print("학습된 장르 목록:", mlb_genre.classes_)

