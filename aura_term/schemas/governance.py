"""
AURA-Term™ governance metadata schemas.

Documents the response structure of ``GET /api/governance`` as returned
by :meth:`~model_engine.BankingModelEngine.get_governance_metadata` with
the ``audit_log_notice`` field appended in ``app.py``.

The current route returns the raw dict from the engine without a declared
``response_model``. These Pydantic models document the contract for future
typed usage in Phase 11+ (React type generation and OpenAPI schema).

IMPORTANT — Accuracy Notices Embedded in Source
------------------------------------------------
``demographic_fairness_parity`` values in the live response are marked
``"status": "Illustrative"`` and must not be presented as validated fairness
measurements against this model's training data.

``compliance_standards`` entries carry ``"notice": "Reference framework — ..."``
disclaimers indicating they are not certifications or regulatory approvals.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class ComplianceStandard(BaseModel):
    """A reference compliance framework entry — NOT a certification."""

    code: str
    scope: str
    notice: str


class FairnessCohort(BaseModel):
    """Illustrative demographic fairness parity entry (sample data only)."""

    cohort: str
    sample_representation: str
    disparate_impact_ratio: str
    status: str


class LayerSpec(BaseModel):
    """Neural network layer specification for governance disclosure."""

    layer: str
    units: Optional[int] = None
    activation: Optional[str] = None
    params: Optional[int] = None
    rate: Optional[float] = None
    shape: Optional[List[Any]] = None
    details: Optional[str] = None


class AuditLogEntry(BaseModel):
    """Single entry from the in-process audit log."""

    request_id: str
    timestamp: str
    client_age: float
    client_job: str
    balance: float
    propensity_score: float
    decision_tier: str
    inference_ms: float


class GovernanceResponse(BaseModel):
    """
    Full governance metadata response returned by ``GET /api/governance``.

    Note: The current ``app.py`` route returns this structure as a raw dict
    without a declared response_model. This Pydantic model documents the
    contract for future typed usage.
    """

    model_name: str
    version: str
    sha256_checksum: str
    target_variable: str
    task_type: str
    intended_use: str
    out_of_scope_use: str
    pre_call_leakage_protection: str
    training_dataset: str
    model_architecture: Dict[str, Any]
    compliance_standards: List[ComplianceStandard]
    demographic_fairness_parity: List[FairnessCohort]
    fairness_data_notice: str
    recent_audit_log: List[AuditLogEntry]
    audit_log_notice: str
