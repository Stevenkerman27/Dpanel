"""CAD-boundary checks for declared trailing-edge input."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "audit" / "runtime" / "python"))
import gmsh
from cad_te import validate_declared_te


class DeclaredTrailingEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = ROOT / "results" / "cad_inputs_v1" / "naca4412_wing"
        cls.te = np.asarray(json.loads(
            (cls.fixture / "trailing_edge.json").read_text(encoding="utf-8"))["points"])
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("declared_te_validation")
        imported = gmsh.model.occ.importShapes(
            str((cls.fixture / "wing.step").resolve()), highestDimOnly=True)
        gmsh.model.occ.synchronize()
        cls.wing_solids = [entity for entity in imported if entity[0] == 3]
        if len(cls.wing_solids) != 1:
            raise AssertionError(f"Expected one wing solid, found {len(cls.wing_solids)}")

    @classmethod
    def tearDownClass(cls):
        gmsh.finalize()

    def test_naca4412_declared_te_matches_trimmed_boundary(self):
        report = validate_declared_te(gmsh, self.wing_solids, self.te, 1e-7)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["all_samples_on_trimmed_wing_boundary"])

    def test_streamwise_offset_is_rejected(self):
        offset_te = self.te.copy()
        offset_te[:, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "trimmed wing boundary curve"):
            validate_declared_te(gmsh, self.wing_solids, offset_te, 1e-7)

    def test_spanwise_extension_is_rejected(self):
        extended_te = self.te.copy()
        extended_te[0, 1] -= 0.05
        with self.assertRaisesRegex(ValueError, "trimmed wing boundary curve"):
            validate_declared_te(gmsh, self.wing_solids, extended_te, 1e-7)


if __name__ == "__main__":
    unittest.main()
