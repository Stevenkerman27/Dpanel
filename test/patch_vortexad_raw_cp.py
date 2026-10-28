"""Expose unclipped Cp in the locked VortexAD checkout; no aerodynamic changes."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=Path, default=Path("runs/backend_probes/src/VortexAD"))
args = parser.parse_args()
path = args.backend / "VortexAD/core/pm/post_processor.py"
source = path.read_text(encoding="utf-8")
if "output_dict['Cp_raw'] = Cp_raw" not in source:
    original = """    Cp_cutoff_exp = csdl.expand(Cp_cutoff, Cp.shape)
    Cp = csdl.maximum(Cp, Cp_cutoff_exp, rho=100)"""
    replacement = """    # Preserve physical Cp before optional numerical clipping.
    Cp_raw = Cp
    if Cp_cutoff is not None:
        Cp_cutoff_exp = csdl.expand(Cp_cutoff, Cp.shape)
        Cp = csdl.maximum(Cp, Cp_cutoff_exp, rho=100)"""
    assert source.count(original) == 1, "Use locked VortexAD ab649398; inspect other revisions separately"
    assert source.count("    output_dict['Cp'] = Cp") == 1
    source = source.replace(original, replacement)
    source = source.replace("    output_dict['Cp'] = Cp", "    output_dict['Cp'] = Cp\n    output_dict['Cp_raw'] = Cp_raw")
    path.write_text(source, encoding="utf-8")
assert "    if Cp_cutoff is not None:" in source
print("Raw Cp exposed; cutoff=None bypasses pressure clipping:", path)
