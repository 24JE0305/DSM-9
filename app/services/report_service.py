# app/services/report_service.py

from datetime import datetime


def compute_prediction_score(predictions: dict) -> float:
    """
    Improved scoring logic.
    """

    p90 = predictions.get("90_days", {})
    expected_return = p90.get("return_percentage", 0)
    confidence = p90.get("confidence_score", 0)
    bias = p90.get("signal_bias", "Neutral")

    # Return scoring (scale up to 30% max)
    return_score = min((expected_return / 30) * 10, 10)

    # Confidence scoring (assuming max ~30)
    confidence_score = min((confidence / 30) * 10, 10)

    # Bias boost
    bias_boost = 0
    if "Strong Bullish" in bias:
        bias_boost = 1
    elif "Bullish" in bias:
        bias_boost = 0.5

    score = (
        0.6 * return_score +
        0.4 * confidence_score +
        bias_boost
    )

    return round(min(score, 10), 2)


def compute_risk_score(risk_metrics: dict) -> float:
    """
    Convert volatility + model agreement into risk score.
    """
    volatility = risk_metrics.get("volatility_level", "High")
    agreement = risk_metrics.get("model_agreement", 0)

    if volatility == "Low":
        base = 9
    elif volatility == "Moderate":
        base = 7
    else:
        base = 5

    agreement_boost = agreement * 2  # scale

    score = base + agreement_boost

    return round(min(score, 10), 2)


def compute_final_rating(prediction_score: float, risk_score: float) -> float:
    """
    Weighted final rating out of 10.
    """
    # Placeholder fundamental score (we'll replace later)
    fundamental_score = 7.0

    final = (
        0.4 * prediction_score +
        0.3 * fundamental_score +
        0.3 * risk_score
    )

    return round(min(final, 10), 2)


def build_report(ticker: str, prediction_data: dict) -> dict:

    predictions = prediction_data["predictions"]
    risk_metrics = prediction_data["risk_metrics"]

    prediction_score = compute_prediction_score(predictions)
    risk_score = compute_risk_score(risk_metrics)
    final_rating = compute_final_rating(prediction_score, risk_score)

    return {
        "symbol": ticker,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_info": prediction_data["model_info"],
        "prediction_summary": predictions,
        "risk_metrics": risk_metrics,
        "rating": {
            "prediction_strength": prediction_score,
            "risk_adjusted_score": risk_score,
            "final_rating_out_of_10": final_rating
        },
        "disclaimer": prediction_data["disclaimer"]
    }