import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import random
from collections import defaultdict

class OTTRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self._prepare_data()

    def _prepare_data(self):
        # NaN 처리 및 벡터화
        self.df['장르'] = self.df['장르'].fillna('').apply(lambda x: x.split(', '))
        self.df['OTT'] = self.df['OTT'].fillna('').apply(lambda x: x.split(', '))

        self.mlb_ott = MultiLabelBinarizer()
        self.mlb_genre = MultiLabelBinarizer()

        self.ott_vec = self.mlb_ott.fit_transform(self.df['OTT'])
        self.genre_vec = self.mlb_genre.fit_transform(self.df['장르'])

        scaler = MinMaxScaler()
        self.rating_vec = scaler.fit_transform(self.df[['평점']].fillna(0))

        self.content_vec = np.hstack([self.ott_vec, self.genre_vec, self.rating_vec])

        # KNN 모델 학습
        self.knn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.knn_model.fit(self.content_vec)

    def recommend(self, user_ott, user_genre, total_needed=10):
        # 사용자 벡터 생성
        user_ott_vec = self.mlb_ott.transform([user_ott])
        user_genre_vec = self.mlb_genre.transform([user_genre])
        user_rating_vec = [[1.0]]
        user_vec = np.hstack([user_ott_vec, user_genre_vec, user_rating_vec])

        # KNN 유사 콘텐츠 찾기
        distances, indices = self.knn_model.kneighbors(user_vec)

        # 유사 콘텐츠 추출
        candidates = self.df.iloc[indices[0]].copy()
        candidates['유사도'] = 1 - distances[0]

        # 유사도 0.7 이상 콘텐츠만 필터
        filtered = candidates[candidates['유사도'] >= 0.75].sample(frac=1).reset_index(drop=True)

        # 장르 분배 균형 설정
        genres_selected = len(user_genre)
        min_per_genre = total_needed // genres_selected
        extra = total_needed % genres_selected
        genre_limit = {genre: min_per_genre + (1 if i < extra else 0) for i, genre in enumerate(user_genre)}

        recommendations = []
        genre_count = defaultdict(int)

        for _, row in filtered.iterrows():
            content_genres = row['장르']
            matched = [g for g in content_genres if g in user_genre]
            random.shuffle(matched)  # 매번 다른 장르 순서
            for genre in matched:
                if genre_count[genre] < genre_limit[genre]:
                    recommendations.append(row)
                    genre_count[genre] += 1
                    break
            if len(recommendations) >= total_needed:
                break

        # 부족할 경우: 나머지를 랜덤으로 채움
        if len(recommendations) < total_needed:
            already = set(r.name for r in recommendations)
            remaining = filtered[~filtered.index.isin(already)].sample(frac=1)
            recommendations.extend(remaining.head(total_needed - len(recommendations)))

        return pd.DataFrame(recommendations)[['제목', '장르', 'OTT', '평점', '유사도']]
