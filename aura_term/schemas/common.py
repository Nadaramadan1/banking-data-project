"""
AURA-Term™ common/shared Pydantic schema models.

Contains shared types used across multiple endpoints, such as error response
envelopes and pagination primitives. Kept minimal in Phase 8A; extended in
later phases when endpoints introduce paginated results or structured errors.
"""
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    """Standard error response body returned by FastAPI on validation failure."""

    detail: str
