"""Service layer — business logic for the prediction endpoint."""
from app.schemas import PlayerInput, PredictionResponse
from src.models.predict import predict_value
from src.utils.serialization import load_model

_artifact = None


def _get_artifact(model_path: str = "models/best_model.joblib") -> dict:
    global _artifact
    if _artifact is None:
        _artifact = load_model(model_path)
    return _artifact


def estimate_player_value(player: PlayerInput) -> PredictionResponse:
    artifact = _get_artifact()
    features = player.model_dump()

    # `potential_gap` n'est PAS injecté ici : le PotentialGapTransformer du pipeline le
    # calcule (source de vérité unique). On ne le dérive que pour l'affichage de la réponse.
    estimated = predict_value(artifact, features)
    potential_gap = player.potential - player.overall

    return PredictionResponse(
        estimated_value_eur=estimated,
        model_name=artifact.get("best_model_name", "unknown"),
        potential_gap=potential_gap,
    )
