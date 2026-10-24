"""Independent algebraic checks for the chosen finite-TE-angle map."""
import json
from pathlib import Path

import numpy as np

theta = np.linspace(.03, 2.*np.pi-.03, 197)
zeta = -.08+1.08*np.exp(1j*theta)
records = []
for exponent in [1.9, 2.]:
    ratio = (zeta-1.)/(zeta+1.)
    mapped = exponent*(1.+ratio**exponent)/(1.-ratio**exponent)
    derivative = 4.*exponent**2*ratio**(exponent-1.)/((zeta+1.)**2*(1.-ratio**exponent)**2)
    step = 1e-6
    ratio_plus = (zeta+step-1.)/(zeta+step+1.)
    ratio_minus = (zeta-step-1.)/(zeta-step+1.)
    plus = exponent*(1.+ratio_plus**exponent)/(1.-ratio_plus**exponent)
    minus = exponent*(1.+ratio_minus**exponent)/(1.-ratio_minus**exponent)
    derivative_error = float(np.max(abs((plus-minus)/(2.*step)-derivative)))
    inverse_ratio = ((mapped-exponent)/(mapped+exponent))**(1./exponent)
    inverse = (1.+inverse_ratio)/(1.-inverse_ratio)
    inverse_error = float(np.max(abs(inverse-zeta)))
    records.append(dict(test="map_derivative_and_inverse", exponent=exponent,
                        derivative_error=derivative_error, inverse_error=inverse_error,
                        passed=derivative_error < 1e-7 and inverse_error < 1e-12))
    if exponent == 2.:
        error = float(np.max(abs(mapped-(zeta+1./zeta))))
        records.append(dict(test="n2_joukowski_limit", error=error, passed=error < 1e-12))
Path(__file__).with_name("mapping_tests.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records, indent=2))
assert all(row["passed"] for row in records)
