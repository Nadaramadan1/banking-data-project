"""
AURA-Term™ Pydantic schemas sub-package.

Modules
-------
common
    Shared response envelope types (pagination, error models).
client
    ClientFeatures request schema and single-prediction response envelope.
batch
    BatchRequest schema for JSON batch scoring.
sensitivity
    Sensitivity curve request/response types.
governance
    Model governance metadata response types.
"""
from aura_term.schemas.client import ClientFeatures
from aura_term.schemas.batch import BatchRequest

__all__ = [
    "ClientFeatures",
    "BatchRequest",
]
