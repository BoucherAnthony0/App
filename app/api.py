"""FastAPI REST API — run: uvicorn app.api:app --reload"""
from fastapi import FastAPI, HTTPException
from app.schemas import PlayerInput, PredictionResponse
from app.service import estimate_player_value

app = FastAPI(
    title="Football Scout API",
    description="Estimates the market value of a professional football player.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(player: PlayerInput) -> PredictionResponse:
    try:
        return estimate_player_value(player)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python src/pipeline.py` first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
