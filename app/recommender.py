import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
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

        scaler = MinMaxScaler()
        self.rating_vec = scaler.fit_transform(self.df[['평점']].fillna(0))
        self.content_vec = np.hstack([self.ott_vec, self.genre_vec, self.rating_vec])

    def recommend(self, user_ott, user_genre, total_needed=10):
        user_ott_vec = self.mlb_ott.transform([user_ott])
        user_genre_vec = self.mlb_genre.transform([user_genre])
        user_rating_vec = [[1.0]]
        user_vec = np.hstack([user_ott_vec, user_genre_vec, user_rating_vec])

        sims = cosine_similarity(user_vec, self.content_vec)[0]
        self.df['유사도'] = sims

        filtered_df = self.df[self.df['유사도'] >= 0.7].sample(frac=1).copy()

        genres_selected = len(user_genre)
        min_per_genre = total_needed // genres_selected
        extra = total_needed % genres_selected
        genre_limit = {genre: min_per_genre + (1 if i < extra else 0) for i, genre in enumerate(user_genre)}

        recommendations = []
        genre_count = defaultdict(int)

        for _, row in filtered_df.iterrows():
            content_genres = row['장르']
            matched = [g for g in content_genres if g in user_genre]
            for genre in matched:
                if genre_count[genre] < genre_limit[genre]:
                    recommendations.append(row)
                    genre_count[genre] += 1
                    break
            if len(recommendations) >= total_needed:
                break

        if len(recommendations) < total_needed:
            already_selected = set(r.name for r in recommendations)
            remaining = filtered_df[~filtered_df.index.isin(already_selected)]
            more = remaining.head(total_needed - len(recommendations)).to_dict(orient='records')
            recommendations.extend(more)

        return pd.DataFrame(recommendations)[['제목', '장르', 'OTT', '평점', '유사도']]
