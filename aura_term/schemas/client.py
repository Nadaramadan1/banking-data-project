"""
AURA-Term™ client feature schema.

Defines the ``ClientFeatures`` Pydantic model representing the 12 raw input
features accepted by the term-deposit propensity scoring API.

Field definitions, validators, defaults, and constraints are reproduced
exactly from the original ``app.py`` to guarantee zero behavioral change.

Note on derived fields
----------------------
``age_group`` and ``balance_status`` are optional fields that the model engine
will auto-derive if absent. They correspond to the two engineered features
added to the 12 raw inputs before preprocessing.

Immutable constraints
---------------------
- Binary subscription threshold: probability >= 0.50
- Tier 1 (High Propensity):    probability >= 0.70
- Tier 2 (Moderate Propensity): 0.45 <= probability < 0.70
- Tier 3 (Low Propensity):     probability < 0.45
These thresholds are enforced in model_engine.py and must not be changed here.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    """
    The 12 raw telemarketing campaign input features for a single retail
    banking client, plus two optional pre-computed derived features.

    All field definitions (constraints, defaults, descriptions) are identical
    to the original schema in ``app.py``.
    """

    age: float = Field(..., ge=18, le=105, description="Client age")
    balance: float = Field(..., description="Average annual balance in EUR/USD")
    pdays: float = Field(
        -1,
        description="Days passed after previous campaign contact (-1 = not contacted)",
    )
    previous: float = Field(
        0, ge=0, description="Number of contacts performed before this campaign"
    )
    job: str = Field(..., description="Job classification")
    marital: str = Field(
        ..., description="Marital status (married, single, divorced)"
    )
    education: str = Field(
        ...,
        description="Education level (primary, secondary, tertiary, unknown)",
    )
    default: str = Field("no", description="Credit in default (no, yes)")
    housing: str = Field(..., description="Housing loan (no, yes)")
    loan: str = Field("no", description="Personal loan (no, yes)")
    month: str = Field("may", description="Last contact month (jan-dec)")
    poutcome: str = Field(
        "unknown",
        description=(
            "Outcome of previous marketing campaign "
            "(unknown, failure, other, success)"
        ),
    )
    # Optional pre-computed derived features; auto-derived by the engine when absent.
    age_group: Optional[str] = None
    balance_status: Optional[str] = None
