"""Retired invalid cut/fixed-circulation experiment; see KUTTA_CONDITION.md section 10.

The former `target + null_space(J) @ z` changed the requested Gamma and left J target
uncancelled. The `nocut` case also kept the cut/wake DOFs, so it was not a no-cut control.
Use the corrected physical-probe study; historical numerical output is not validation.
"""

if __name__ == "__main__":
    raise SystemExit(
        "Retired invalid experiment. Run: python -m tools.lifting_line_study "
        "--case fixed-gamma --out runs/verification/lifting_line_fix/fixed_gamma")
