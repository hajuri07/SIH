"""
SHAP Explainability Router (Arena 3).
"""

from fastapi import APIRouter, HTTPException
import pandas as pd

from ..schemas import ExplainResponse, SHAPContribution
from ..services import registry
from ..db import get_recent_telemetry

router = APIRouter(prefix="/explain", tags=["Model Explainability"])


@router.get("/{unit_nr}", response_model=ExplainResponse)
async def explain_engine_prediction(unit_nr: int):
    """
    Computes real-time TreeSHAP sensor attributions for the given engine unit.
    Explains which physical sensors are actively driving the RUL reduction or early alert.
    """
    if not registry.is_loaded:
        registry.load_models()

    if not registry.shap_explainer or not registry.feature_engineer:
        raise HTTPException(
            status_code=503,
            detail="Explainability engine not initialized. Run training pipeline first."
        )

    records = await get_recent_telemetry(unit_nr, limit=30)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry found for Engine {unit_nr} to explain."
        )

    df_recent = pd.DataFrame(records)
    sensor_cols = [
        c for c in df_recent.columns
        if (c.startswith("s_") or c.startswith("setting_"))
        and not ("_roll_" in c or "_diff_" in c)
    ]

    try:
        transformed_df = registry.feature_engineer.transform(df_recent, sensor_cols)
        latest_row = transformed_df.iloc[-1]
        cycle = int(latest_row["time_cycles"])

        rul_pred = float(registry.xgboost_model.predict(transformed_df.iloc[[-1]])[0])
        explanation = registry.shap_explainer.explain_instance(latest_row, top_k=8)

        contributions = [
            SHAPContribution(
                feature=c["feature"],
                feature_value=c["feature_value"],
                shap_value=c["shap_value"],
                impact=c["impact"]
            )
            for c in explanation["top_contributions"]
        ]

        top_driver = contributions[0].feature if contributions else "Unknown"
        summary = (
            f"Engine {unit_nr} at cycle {cycle} predicted RUL: {round(rul_pred, 1)} cycles. "
            f"Primary degradation driver is {top_driver} with SHAP contribution of {contributions[0].shap_value}."
        )

        return ExplainResponse(
            unit_nr=unit_nr,
            cycle=cycle,
            predicted_rul=round(rul_pred, 1),
            base_value=explanation["base_value"],
            top_contributions=contributions,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
