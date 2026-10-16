"""
AURA-Term™ sensitivity analysis schemas.

The sensitivity endpoint (``POST /api/sensitivity``) accepts a
``ClientFeatures`` payload and returns three response curves computed by
:meth:`~model_engine.BankingModelEngine.generate_sensitivity_curves`:

- **balance_curve**: propensity scores across balance values €0–€100,000.
- **age_curve**: propensity scores across client ages 20–80.
- **month_curve**: propensity scores for each contact month (jan–dec).

The raw dict returned by ``generate_sensitivity_curves`` is serialized
directly as the HTTP response body — no response model wrapper is declared
in the current ``app.py`` route.  This module provides typed documentation
of that structure for use in Phase 11+ (React type generation).
"""
from typing import Dict, List, Union
from pydantic import BaseModel


class SensitivityCurve(BaseModel):
    """A single sensitivity response curve with x-axis labels and y-axis scores."""

    x: List[Union[float, int, str]]
    y: List[float]


class SensitivityCurveResponse(BaseModel):
    """
    Full sensitivity analysis response returned by ``POST /api/sensitivity``.

    Note: The current ``app.py`` route returns this structure directly from
    ``engine.generate_sensitivity_curves()`` without a declared response_model.
    This Pydantic model documents the contract for future typed usage.
    """

    balance_curve: SensitivityCurve
    age_curve: SensitivityCurve
    month_curve: SensitivityCurve
