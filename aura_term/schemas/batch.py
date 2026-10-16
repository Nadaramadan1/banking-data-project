"""
AURA-Term™ batch scoring schemas.

Defines the ``BatchRequest`` model for JSON batch scoring requests sent to
``POST /api/batch``. The ``accounts`` field contains a list of
:class:`~aura_term.schemas.client.ClientFeatures` instances.

The existing route in ``app.py`` validates ``len(accounts) <= settings.max_batch_rows``
(default: 5,000 records) and rejects empty lists with HTTP 400.
"""
from typing import List
from pydantic import BaseModel
from aura_term.schemas.client import ClientFeatures


class BatchRequest(BaseModel):
    """
    JSON batch scoring request body.

    Attributes
    ----------
    accounts:
        Ordered list of client feature records to score. Must be non-empty
        and not exceed ``settings.max_batch_rows`` (default: 5,000).
    """

    accounts: List[ClientFeatures]
