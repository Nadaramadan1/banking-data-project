"""
AURA-Term Backend Settings — Backward-Compatible Facade
========================================================
This module is retained as the public import path for legacy callers:

    from settings import settings          # application singleton
    from settings import settings, Settings  # both class and singleton

The canonical implementation has moved to ``aura_term.core.config``.
This file re-exports ``Settings`` and ``settings`` from there so that
no existing import path requires modification.

Do NOT add logic here. All configuration code belongs in
``aura_term/core/config.py``.
"""
# Re-export the canonical Settings class and the module-level singleton so
# that existing callers such as app.py and the test suite continue to work
# without modification:
#
#   from settings import settings          -> aura_term.core.config.settings
#   from settings import settings, Settings -> both re-exported here
from aura_term.core.config import Settings, settings  # noqa: F401

__all__ = ["Settings", "settings"]
