import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "GRASS-plane" / "cst_airfoil_codec.py").is_file(),
    reason="GRASS-plane CST codec is not installed alongside this repository")


def test_four_section_optimization_records_production_comparison(tmp_path):
    out = tmp_path / "optimization"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "-m", "tools.grass_section_optimize",
                    "--out", str(out), "--iterations", "1"], env=env, check=True)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert len(result["sections_initial"]["chord"]) == 4
    assert len(result["sections_final"]["cst"]) == 4
    assert result["panel_count"] == 156
    assert result["initial"]["CDi"] == pytest.approx(result["formal_initial"]["CDi"], abs=2e-9)
    assert result["final"]["CDi"] == pytest.approx(result["formal_final"]["CDi"], abs=2e-9)
    assert result["physical_accuracy_accepted"] is False
