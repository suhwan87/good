import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# 데이터 로딩 및 전처리
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

# ✅ 기본 추천 함수
def recommend_basic(user_ott, user_genre, total_needed=5, prefer_new=False):
    # 입력 벡터 생성
    user_ott_vec = mlb_ott.transform([user_ott])
    user_genre_vec = mlb_genre.transform([user_genre])
    user_year_vec = [[1.0 if prefer_new else 0.0]]
    user_vec = np.hstack([user_ott_vec, user_genre_vec, user_year_vec])

    # 콘텐츠 벡터에서도 동일한 구조
    sims = cosine_similarity(user_vec, content_vec_initial)[0]

    # 👉 가중치 설정
    ott_weight = 0.2
    genre_weight = 0.6
    year_weight = 0.2 if prefer_new else 0.0

    # 콘텐츠 벡터도 ott + genre + year 순서니까 인덱스 잘라서 따로 계산
    ott_len = user_ott_vec.shape[1]
    genre_len = user_genre_vec.shape[1]

    ott_part = cosine_similarity(user_ott_vec, content_vec_initial[:, :ott_len])[0]
    genre_part = cosine_similarity(user_genre_vec, content_vec_initial[:, ott_len:ott_len + genre_len])[0]
    year_part = content_vec_initial[:, -1].toarray().flatten() if hasattr(content_vec_initial[:, -1], 'toarray') else content_vec_initial[:, -1]

    # 종합 유사도 계산
    sims = (ott_part * ott_weight) + (genre_part * genre_weight)
    if prefer_new:
        sims += year_part * year_weight

    df['유사도'] = sims
    filtered = df[df['유사도'] > 0].sort_values(by='유사도', ascending=False).head(50)
    sampled = filtered.sample(n=min(total_needed, len(filtered)), replace=False)

    return sampled[[
        'CONTENTS_TITLE', 'CONTENTS_GENRE', 'DIRECTOR', 'CAST',
        'OTT', 'RELEASE_YEAR', 'POSTER_IMG', '유사도'
    ]].to_dict(orient='records')


# ✅ 선택 콘텐츠 기반 추천 함수
def recommend_selected(selected_title, total_needed=5):
    if selected_title not in df['CONTENTS_TITLE'].values:
        return []

    idx = df[df['CONTENTS_TITLE'] == selected_title].index[0]
    sims = cosine_similarity(content_vec_detailed[idx], content_vec_detailed).flatten()
    df['유사도'] = sims
    filtered = df[(df.index != idx) & (df['유사도'] > 0)].sort_values(by='유사도', ascending=False).head(50)
    sampled = filtered.sample(n=min(total_needed, len(filtered)), replace=False)

    return sampled[[
        'CONTENTS_TITLE', 'CONTENTS_GENRE', 'DIRECTOR', 'CAST',
        'OTT', 'RELEASE_YEAR', 'POSTER_IMG', '유사도'
    ]].to_dict(orient='records')
