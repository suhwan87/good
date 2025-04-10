from fastapi import FastAPI
from app.api import router
import warnings
from contextlib import asynccontextmanager
import threading
import schedule
import time
import requests
import datetime

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    
warnings.filterwarnings("ignore")

# ✅ keep-alive 스케줄러 정의
def send_ping():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    url = "https://good-vlmd.onrender.com/ping"
    try:
        print(f"[{now}] Sending ping to {url}")
        res = requests.get(url)
        print(f"[{now}] Ping OK: {res.status_code}")
    except Exception as e:
        print(f"[{now}] Ping failed: {e}")

def run_scheduler():
    schedule.every(14).minutes.do(send_ping)
    while True:
        schedule.run_pending()
        time.sleep(1)

# ✅ lifespan 이벤트로 scheduler 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=run_scheduler, daemon=True).start()
    print("✅ keep-alive scheduler started.")
    yield
    print("🛑 App shutting down...")

# ✅ FastAPI 앱 정의 및 router 등록
app = FastAPI(lifespan=lifespan)
app.include_router(router)

# ✅ ping 엔드포인트
@app.get("/ping")
def ping():
    return {"message": "pong"}

# ✅ 루트 엔드포인트
@app.get("/")
async def index():
    return {
        "message": "콘텐츠 추천 API 작동 중!",
        "how_to_use": "/recommend/basic 또는 /recommend/selected를 확인하세요."
    }