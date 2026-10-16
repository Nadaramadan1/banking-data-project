"""Phase 8A smoke tests for import compatibility and schema validation."""
import sys

errors = []

# Test 1: Legacy import path
try:
    from settings import settings, Settings
    assert hasattr(settings, 'environment')
    assert hasattr(settings, 'max_batch_rows')
    assert hasattr(settings, 'upload_chunk_size')
    assert callable(settings.validate_on_startup)
    print('[PASS] from settings import settings, Settings')
except Exception as e:
    errors.append(f'[FAIL] settings legacy import: {e}')

# Test 2: Canonical path
try:
    from aura_term.core.config import settings as s2, Settings as S2
    print('[PASS] from aura_term.core.config import settings, Settings')
    assert type(settings).__name__ == 'Settings'
    print(f'       Settings class identity match: {Settings is S2}')
except Exception as e:
    errors.append(f'[FAIL] aura_term.core.config import: {e}')

# Test 3: validate_on_startup still callable
try:
    from settings import settings
    settings.validate_on_startup()
    print('[PASS] settings.validate_on_startup() callable')
except Exception as e:
    errors.append(f'[FAIL] validate_on_startup: {e}')

# Test 4: Schema imports from package
try:
    from aura_term.schemas import ClientFeatures, BatchRequest
    print('[PASS] from aura_term.schemas import ClientFeatures, BatchRequest')
except Exception as e:
    errors.append(f'[FAIL] aura_term.schemas import: {e}')

# Test 5: ClientFeatures defaults
try:
    from aura_term.schemas.client import ClientFeatures
    cf = ClientFeatures(age=42, balance=2500.0, job='technician',
                        marital='married', education='secondary', housing='yes')
    d = cf.model_dump()
    assert d['pdays'] == -1.0
    assert d['previous'] == 0.0
    assert d['default'] == 'no'
    assert d['loan'] == 'no'
    assert d['month'] == 'may'
    assert d['poutcome'] == 'unknown'
    assert d['age_group'] is None
    assert d['balance_status'] is None
    print('[PASS] ClientFeatures defaults and serialization correct')
    print(f'       keys: {sorted(d.keys())}')
except Exception as e:
    errors.append(f'[FAIL] ClientFeatures defaults: {e}')

# Test 6: ClientFeatures age constraint enforced
try:
    from aura_term.schemas.client import ClientFeatures
    try:
        ClientFeatures(age=10, balance=1000.0, job='x', marital='x', education='x', housing='x')
        errors.append('[FAIL] ClientFeatures should reject age=10')
    except Exception:
        print('[PASS] ClientFeatures rejects age=10 (ValidationError raised)')
except Exception as e:
    errors.append(f'[FAIL] age constraint test error: {e}')

# Test 7: BatchRequest
try:
    from aura_term.schemas.batch import BatchRequest
    from aura_term.schemas.client import ClientFeatures
    cf = ClientFeatures(age=35, balance=1000.0, job='admin',
                        marital='single', education='tertiary', housing='no')
    br = BatchRequest(accounts=[cf])
    assert len(br.accounts) == 1
    assert br.accounts[0].age == 35.0
    print('[PASS] BatchRequest wraps ClientFeatures correctly')
except Exception as e:
    errors.append(f'[FAIL] BatchRequest: {e}')

# Test 8: Sensitivity and governance schema imports
try:
    from aura_term.schemas.sensitivity import SensitivityCurve, SensitivityCurveResponse
    from aura_term.schemas.governance import GovernanceResponse
    from aura_term.schemas.common import ErrorDetail
    print('[PASS] Sensitivity, governance, and common schemas importable')
except Exception as e:
    errors.append(f'[FAIL] extra schema imports: {e}')

# Summary
print()
if errors:
    print('FAILURES:')
    for err in errors:
        print(' ', err)
    sys.exit(1)
else:
    print('All Phase 8A smoke tests PASSED.')
