"""Apply a minimal, reversible local diagnostic patch to VortexAD source.

This is not a vendored upstream fork. It preserves Cp_raw before clipping and
makes Cp_cutoff=None disable clipping in the unstructured post-processor.
Usage:
  python test/patch_vortexad_raw_cp.py runs/backend_probes/src/VortexAD
"""
from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
root = Path(sys.argv[1]).resolve()
path = root / "VortexAD/core/pm/post_processor.py"
text = path.read_text(encoding="utf-8")
old = """    Cp_cutoff_exp = csdl.expand(Cp_cutoff, Cp.shape)
    Cp = csdl.maximum(Cp, Cp_cutoff_exp, rho=100)
"""
new = """    # Preserve physical Cp before optional numerical clipping.
    Cp_raw = Cp
    if Cp_cutoff is not None:
        Cp_cutoff_exp = csdl.expand(Cp_cutoff, Cp.shape)
        Cp = csdl.maximum(Cp, Cp_cutoff_exp, rho=100)
"""
old_out = """    output_dict['Cp'] = Cp
    output_dict['panel_forces'] = dF
"""
new_out = """    output_dict['Cp'] = Cp
    output_dict['Cp_raw'] = Cp_raw
    output_dict['panel_forces'] = dF
"""
if new in text and new_out in text:
    print("already patched:", path)
    raise SystemExit(0)
if text.count(old) != 1 or text.count(old_out) != 1:
    raise SystemExit(
        "Expected exact upstream snippets once each; source version mismatch. "
        "Do not patch blindly. Inspect " + str(path)
    )
text = text.replace(old, new).replace(old_out, new_out)
path.write_text(text, encoding="utf-8")
print("patched local diagnostic checkout:", path)
