"""VSPAERO input file (.vspaero) writer.

The field order and formatting follow the files that produced the numbers in
`docs/VSP_TRAPEZOID_CONVERGENCE.md`; `to_text()` is byte-stable so a round
trip through `SweepConfig.from_text()` reproduces an existing case file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional, Sequence

_WAKE_DEFAULTS = dict(
    wake_iters=20,
    wake_nodes=32,
    wake_relax=0.5,
    far_dist=5.0,
)


@dataclass
class SweepConfig:
    """Everything the `.vspaero` file carries.

    Defaults are the settings used for the thick-panel ladder
    (`wake_relax=0.5`, `wake_iters=20`, `wake_nodes=32`, `far_dist=5`), which
    is the *conservative* choice; the user's GUI model ran relax=1 / iters=5 /
    nodes=8 and produced the same `CLwtot` to 1%.
    """

    sref: float = 0.30
    cref: float = 0.15
    bref: float = 2.0
    x_cg: float = 0.0
    y_cg: float = 0.0
    z_cg: float = 0.0
    mach: float = 0.0
    alpha: List[float] = field(default_factory=lambda: [0.0])
    beta: float = 0.0
    rec_ref: float = 1.0e7
    vinf: float = 100.0
    rho: float = 0.002377
    symmetry: int = 1
    wake_iters: int = 20
    wake_nodes: int = 32
    wake_relax: float = 0.5
    far_toggle: int = 1
    far_dist: float = 5.0
    implicit_wake: int = 0
    update_preconditioner: int = 0
    tecplot: int = 0
    stall_model: int = 0

    def __post_init__(self) -> None:
        self.alpha = [float(a) for a in self.alpha]
        if not self.alpha:
            raise ValueError("alpha list must not be empty")
        if self.wake_iters <= 0:
            raise ValueError("wake_iters must be positive")
        if self.sref <= 0 or self.cref <= 0 or self.bref <= 0:
            raise ValueError("sref, cref and bref must be positive")

    # ------------------------------------------------------------------ text
    def to_text(self) -> str:
        aoa = ", ".join("%.10g" % a for a in self.alpha)
        lines = [
            "Sref = %.17g" % self.sref,
            "Cref = %.17g" % self.cref,
            "Bref = %.17g" % self.bref,
            "X_cg = %g" % self.x_cg,
            "Y_cg = %g" % self.y_cg,
            "Z_cg = %g" % self.z_cg,
            "Mach = %g" % self.mach,
            "AoA = " + aoa,
            "Beta = %g" % self.beta,
            "ReCref = %.17g" % self.rec_ref,
            "Vinf = %g" % self.vinf,
            "Rho = %g" % self.rho,
            "StallModel = %d" % self.stall_model,
            "Clo2D = 0",
            "CLMax2D = 1",
            "Symmetry = %d" % self.symmetry,
            "FreezeMultiPoleAtIteration = 10000",
            "FreezeWakeAtIteration = 10000",
            "FreezeWakeRootVortices = 0",
            "ImplicitWake = %d" % self.implicit_wake,
            "ImplicitWakeStartIteration = 0",
            "FarDistToggle = %d" % self.far_toggle,
            "FarDist = %.17g" % self.far_dist,
            "NumWakeNodes = %d" % self.wake_nodes,
            "WakeIters = %d" % self.wake_iters,
            "WakeRelax = %.17g" % self.wake_relax,
            "ForwardGMRESConvergenceFactor = 1",
            "AdjointGMRESConvergenceFactor = 1",
            "NonLinearConvergenceFactor = 1",
            "CoreSizeFactor = 1",
            "FarAway = 5",
            "UpdateMatrixPreconditioner = %d" % self.update_preconditioner,
            "UseWakeNodeMatrixPreconditioner = 0",
            "Write2DFEMFile = 0",
            "WriteTecplotFile = %d" % self.tecplot,
            "NumberOfControlGroups = 0",
            "NumberofSurveyPoints = 0",
            "QuadTreeBufferLevels = 0",
            "NumberOfQuadTrees = 1",
            "1 2 0",
            "NumberOfInlets = 0",
            "NumberOfNozzles = 0",
            "VSP_StabilityType = 0",
        ]
        return "\r\n".join(lines) + "\r\n"

    def write(self, path) -> None:
        from pathlib import Path

        Path(path).write_text(self.to_text(), encoding="utf-8", newline="")

    # ------------------------------------------------------------------ read
    @classmethod
    def from_text(cls, text: str) -> "SweepConfig":
        values = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            raw = raw.strip()
            if key == "AoA":
                values["alpha"] = [float(v) for v in raw.replace(",", " ").split()]
            elif key in _MAP:
                try:
                    values[_MAP[key]] = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
                except ValueError:
                    continue
        cfg = cls(**values)
        return cfg

    def as_dict(self) -> dict:
        return asdict(self)


# .vspaero key -> dataclass field
_MAP = {
    "Sref": "sref",
    "Cref": "cref",
    "Bref": "bref",
    "X_cg": "x_cg",
    "Y_cg": "y_cg",
    "Z_cg": "z_cg",
    "Mach": "mach",
    "Beta": "beta",
    "ReCref": "rec_ref",
    "Vinf": "vinf",
    "Rho": "rho",
    "Symmetry": "symmetry",
    "WakeIters": "wake_iters",
    "NumWakeNodes": "wake_nodes",
    "WakeRelax": "wake_relax",
    "FarDistToggle": "far_toggle",
    "FarDist": "far_dist",
    "ImplicitWake": "implicit_wake",
    "UpdateMatrixPreconditioner": "update_preconditioner",
    "WriteTecplotFile": "tecplot",
    "StallModel": "stall_model",
}
