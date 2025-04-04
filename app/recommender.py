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
        self.df['장르'] = self.df['장르'].fillna('').apply(lambda x: x.split(', '))
        self.df['OTT'] = self.df['OTT'].fillna('').apply(lambda x: x.split(', '))

        self.mlb_ott = MultiLabelBinarizer()
        self.mlb_genre = MultiLabelBinarizer()

        self.ott_vec = self.mlb_ott.fit_transform(self.df['OTT'])
        self.genre_vec = self.mlb_genre.fit_transform(self.df['장르'])

        current_year = 2025
        self.df['제작연도'] = self.df['제작연도'].fillna('2000년')
        self.df['year_score'] = self.df['제작연도'].apply(
            lambda y: max(0, 25 - (current_year - int(str(y)[:4])))
        )
        scaler = MinMaxScaler()
        self.year_vec = scaler.fit_transform(self.df[['year_score']])

    def recommend(self, user_ott, user_genre, total_needed=10, latest_only=False):
        user_ott_vec = self.mlb_ott.transform([user_ott])
        user_genre_vec = self.mlb_genre.transform([user_genre])

        if latest_only:
            user_year_vec = [[1.0]]
            user_vec = np.hstack([user_ott_vec, user_genre_vec, user_year_vec])
            content_vec = np.hstack([self.ott_vec, self.genre_vec, self.year_vec])
        else:
            user_vec = np.hstack([user_ott_vec, user_genre_vec])
            content_vec = np.hstack([self.ott_vec, self.genre_vec])

        knn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
        knn_model.fit(content_vec)
        distances, indices = knn_model.kneighbors(user_vec)

        candidates = self.df.iloc[indices[0]].copy()
        candidates['유사도'] = 1 - distances[0]

        filtered = candidates[candidates['유사도'] >= 0.7].sample(frac=1).reset_index(drop=True)

        genres_selected = len(user_genre)
        min_per_genre = total_needed // genres_selected
        extra = total_needed % genres_selected
        genre_limit = {genre: min_per_genre + (1 if i < extra else 0) for i, genre in enumerate(user_genre)}

        recommendations = []
        genre_count = defaultdict(int)

        for _, row in filtered.iterrows():
            content_genres = row['장르']
            matched = [g for g in content_genres if g in user_genre]
            random.shuffle(matched)
            for genre in matched:
                if genre_count[genre] < genre_limit[genre]:
                    recommendations.append(row)
                    genre_count[genre] += 1
                    break
            if len(recommendations) >= total_needed:
                break

        if len(recommendations) < total_needed:
            already = set(r.name for r in recommendations)
            remaining = filtered[~filtered.index.isin(already)].sample(frac=1)
            recommendations.extend(remaining.head(total_needed - len(recommendations)))

        return pd.DataFrame(recommendations)[['제목', '장르', 'OTT', '평점', '제작연도', '유사도']]