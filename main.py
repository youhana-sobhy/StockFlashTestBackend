import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from engine import run_backtest
from schemas import BacktestRequest, BacktestResponse

app = FastAPI(
    title="StockFlashTest Backtesting API",
    description="Backtesting as Code platform powered by FastAPI and Pandas",
    version="1.0.0",
)


DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://www.sincer.info",
    "https://sincer.info",
]


def get_allowed_origins() -> list[str]:
    """
    Build the CORS allowlist from defaults, FRONTEND_URL, and FRONTEND_URLS.
    FRONTEND_URL should be the full production URL, e.g. https://your-app.vercel.app
    FRONTEND_URLS can provide additional comma-separated origins if needed.
    """
    origins = list(DEFAULT_ALLOWED_ORIGINS)

    frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
    if frontend_url and frontend_url != "*" and frontend_url not in origins:
        origins.append(frontend_url)

    extra_origins = os.getenv("FRONTEND_URLS", "")
    for origin in extra_origins.split(","):
        cleaned = origin.strip().rstrip("/")
        if cleaned and cleaned not in origins:
            origins.append(cleaned)

    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> BacktestResponse:
    try:
        return run_backtest(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {exc}") from exc
