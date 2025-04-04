import requests
import pandas as pd

# 👉 Render에 배포된 API의 기본 URL을 넣어줘
BASE_URL = "https://good-vlmd.onrender.com"

# ✅ 테스트할 사용자 입력 값
params = {
    "user_ott": ["넷플릭스", "왓챠", "티빙"],
    "user_genre": ["로맨스", "스릴러", "액션", "코미디", "공포", "판타지"],
    "total_needed": 10,
    "latest_only": True  # ← 최신 작품만 추천받고 싶다면 추가
}

# 🔍 추천 API 호출
response = requests.get(f"{BASE_URL}/recommend", params=params)

# ✅ 결과 확인
if response.status_code == 200:
    results = response.json()
    df = pd.DataFrame(results)
    print(df[['제목', '장르', 'OTT', '평점', '제작연도', '유사도']])
else:
    print("❌ API 호출 실패:", response.status_code)
    print(response.text)
