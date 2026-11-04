"""Parametric wing definition for the browser workbench (GRASS-style variables).

All lengths are metres, all angles degrees. The browser builds its parameter
form from SCHEMA, so this module is the single declaration of what the user may
edit. Values are clamped to their declared range; nothing else is silently
"corrected", so a typo shows up as a clamp the log can name.

CAD contract: the declared trailing edge must lie on the exported wall (and,
since issue #100, it no longer has to lie in z=0). The trailing edge is the
datum of every section: the airfoil is rotated about its own TE point by the
local twist, so a section never moves its own TE. Dihedral is applied as a pure
z translation of the whole section, `z_te = |y| * tan(dihedral)`, which keeps
the root TE on z=0 (the symmetry plane) and makes the declared TE a 3-D
polyline; `panel/cad_half.py` builds its wake strips with that same z, so the
sheet leaves the wing along the real TE instead of a flat z=0 line.
"""
from __future__ import annotations

import math

SCHEMA = [
    dict(key="origin_x", label="根前缘 x（绝对）", unit="m", default=0.0, min=-50.0, max=50.0, step=0.1),
    dict(key="span", label="全展长 b", unit="m", default=6.0, min=0.5, max=30.0, step=0.1),
    dict(key="root_chord", label="根弦长", unit="m", default=1.0, min=0.1, max=5.0, step=0.05),
    dict(key="tip_chord", label="梢弦长", unit="m", default=0.5, min=0.05, max=5.0, step=0.05),
    dict(key="sweep_deg", label="前缘后掠", unit="deg", default=12.0, min=-45.0, max=60.0, step=1.0),
    dict(key="dihedral_deg", label="上反角", unit="deg", default=0.0, min=-20.0, max=20.0, step=0.5),
    dict(key="twist_root_deg", label="根部扭转", unit="deg", default=2.0, min=-8.0, max=8.0, step=0.5),
    dict(key="twist_tip_deg", label="梢部扭转", unit="deg", default=-1.0, min=-8.0, max=8.0, step=0.5),
    dict(key="thickness_root", label="根部厚度比", unit="-", default=0.12, min=0.03, max=0.25, step=0.005),
    dict(key="thickness_tip", label="梢部厚度比", unit="-", default=0.10, min=0.03, max=0.25, step=0.005),
    dict(key="camber_root", label="根部弯度", unit="-", default=0.02, min=0.0, max=0.10, step=0.005),
    dict(key="camber_tip", label="梢部弯度", unit="-", default=0.02, min=0.0, max=0.10, step=0.005),
    dict(key="camber_position", label="最大弯度位置", unit="-", default=0.4, min=0.2, max=0.7, step=0.01),
    dict(key="sections", label="展向站数", unit="-", default=8, min=3, max=24, step=1),
]
INT_KEYS = frozenset(["sections"])

# Non-lifting bodies (fuselages) are separate parts; panel/cad_half.py takes
# them through --body-cad and treats their wall faces as wetted but not lifting,
# so any number may be declared next to the single lifting surface.
BODY_SCHEMA = [
    dict(key="length", label="机身长度", unit="m", default=3.0, min=0.5, max=20.0, step=0.1),
    dict(key="diameter", label="最大直径", unit="m", default=0.6, min=0.05, max=5.0, step=0.05),
    dict(key="center_x", label="中心 x（绝对）", unit="m", default=0.7, min=-50.0, max=50.0, step=0.1),
    dict(key="center_z", label="中心 z（绝对）", unit="m", default=-0.45, min=-20.0, max=20.0, step=0.1),
]


def normalise_body(raw):
    """Fill and clamp one fuselage part; the body is axisymmetric about y=0.

    center_x / center_z are absolute coordinates. The old nose_x key is still
    accepted and converted, so saved payloads keep working.
    """
    raw = dict(raw or {})
    if "center_x" not in raw and "nose_x" in raw:
        raw["center_x"] = float(raw["nose_x"]) + 0.5 * float(raw.get("length", 3.0))
    out = {}
    clamped = []
    for item in BODY_SCHEMA:
        key = item["key"]
        try:
            number = float(raw.get(key, item["default"]))
        except (TypeError, ValueError):
            raise ValueError("机身参数 %s 不是数字: %r" % (key, raw.get(key)))
        if not math.isfinite(number):
            raise ValueError("机身参数 %s 不是有限数: %r" % (key, raw.get(key)))
        lo, hi = float(item["min"]), float(item["max"])
        if number < lo or number > hi:
            clamped.append(key)
            number = min(max(number, lo), hi)
        out[key] = number
    return out, clamped


def normalise(raw):
    """Return the parameter dictionary with defaults filled and values clamped."""
    raw = raw or {}
    out = {}
    clamped = []
    for item in SCHEMA:
        key = item["key"]
        value = raw.get(key, item["default"])
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("参数 %s 不是数字: %r" % (key, raw.get(key)))
        if not math.isfinite(number):
            raise ValueError("参数 %s 不是有限数: %r" % (key, raw.get(key)))
        lo, hi = float(item["min"]), float(item["max"])
        if number < lo or number > hi:
            clamped.append(key)
            number = min(max(number, lo), hi)
        out[key] = number
    for key in INT_KEYS:
        out[key] = int(round(out[key]))
    if out["tip_chord"] > out["root_chord"] * 3.0:
        raise ValueError("梢弦长不能超过根弦长的三倍（平面形状会退化）")
    if out["sections"] < 3:
        raise ValueError("展向站数至少为 3")
    return out, clamped


def _lerp(a, b, t):
    return a + (b - a) * t


def half_stations(p):
    """Span stations from the symmetry plane (y=0) to the tip (y=b/2).

    `te_x` is the trailing-edge x of the station and `te_z` its height above
    the symmetry plane; `te_z = |y| tan(dihedral)`, so a station's TE moves as a
    rigid body and the root TE stays at z=0.
    """
    n = int(p["sections"])
    half = 0.5 * float(p["span"])
    sweep = math.tan(math.radians(p["sweep_deg"]))
    dihedral = math.tan(math.radians(float(p.get("dihedral_deg", 0.0))))
    out = []
    for i in range(n):
        f = i / (n - 1.0)
        y = half * f
        chord = _lerp(p["root_chord"], p["tip_chord"], f)
        le_x = float(p.get("origin_x", 0.0)) + y * sweep
        te_z = abs(y) * dihedral
        twist = math.radians(_lerp(p["twist_root_deg"], p["twist_tip_deg"], f))
        out.append(dict(y=y, chord=chord, le_x=le_x, te_x=le_x + chord, te_z=te_z,
                        dihedral=dihedral, twist=twist,
                        tc=_lerp(p["thickness_root"], p["thickness_tip"], f),
                        camber=_lerp(p["camber_root"], p["camber_tip"], f),
                        position=p["camber_position"]))
    return out


def full_stations(p):
    """Full-span stations, ascending in y, mirrored about the symmetry plane."""
    half = half_stations(p)
    out = []
    for s in reversed(half):
        if s["y"] <= 0.0:
            continue
        # The two halves must roll in opposite directions about x, otherwise the
        # dihedral wing is not symmetric about y=0 and cad_half rejects its y
        # extent. Twist and te_z stay as they are.
        out.append(dict(s, y=-s["y"], dihedral=-s["dihedral"]))
    out.extend(half)
    return out


def trailing_edge_points(p):
    """Declared trailing edge across the full span (y ascending, 3-D)."""
    return [[float(s["te_x"]), float(s["y"]), float(s["te_z"])] for s in full_stations(p)]


def reference(p):
    """Reference area / chord / moment point declared to the solver."""
    mean_chord = 0.5 * (p["root_chord"] + p["tip_chord"])
    return dict(area=float(mean_chord * p["span"]),
                chord=float(p["root_chord"]),
                point=[float(p.get("origin_x", 0.0) + 0.25 * p["root_chord"]), 0.0, 0.0])
