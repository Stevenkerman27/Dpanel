"""Build DPanel lifting wings from GRASS-plane's CST thick-wing construction.

The surface topology, the Kulfan/CST section codec and the planform frame all
come from D:/3D/Projects/GRASS-plane (panel3d_geometry.thick_wing_topology /
thick_wing_half_mesh, cst_airfoil_codec, wing_geometry_encoder).  The result is
converted into panel.p1_lifting_g1.Wing so it can enter the P1 lifting solve and
the coupled material-wake update unchanged.

Only connectivity/DOF bookkeeping is re-implemented here, following the
conventions of panel.p1_lifting_g1.make_wing: shared LE/TE geometry vertices,
one duplicated potential unknown per interior TE station (lower bank), tip
Gamma prescribed zero, wake sheet extruded downstream in +x.
"""
from __future__ import annotations

import numpy as np

GRASS = "D:/3D/Projects/GRASS-plane"


def grass_modules():
    import sys
    if GRASS not in sys.path:
        sys.path.insert(0, GRASS)
    import cst_airfoil_codec as cst
    import panel3d_geometry as g3d
    return cst, g3d


def naca_code(number: str):
    """CST code for a NACA 4-digit section in GRASS-plane's 100-point layout.

    Layout required by cst_airfoil_codec._validate_target_points: 50 upper
    points from the trailing edge towards the leading edge, the shared leading
    edge at index 50, and 49 lower points back to the trailing edge.
    """
    import torch
    cst, _ = grass_modules()
    m = int(number[0]) / 100.0
    p = int(number[1]) / 10.0 if number[1] != "0" else 0.4
    t = int(number[2:]) / 100.0

    def surfaces(x):
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 - 0.1036 * x ** 4)
        if m == 0:
            return yt, -yt
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2),
                      m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dy = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
        th = np.arctan(dy)
        return yc + yt * np.cos(th), yc - yt * np.cos(th)

    xu = 0.5 * (1 - np.cos(np.linspace(np.pi, 0.02, 50)))
    xl = 0.5 * (1 - np.cos(np.linspace(0.02, np.pi, 49)))
    yu, _ = surfaces(xu)
    _, yl = surfaces(xl)
    pts = torch.as_tensor(np.concatenate([
        np.stack([xu, yu], 1), np.zeros((1, 2)), np.stack([xl, yl], 1)]), dtype=torch.float64)
    fitted = cst.fit_airfoil_points(pts)
    return torch.as_tensor(fitted["code"], dtype=torch.float64).reshape(1, -1)


def dat_code(name: str):
    """CST code for a Selig .dat section, resampled to GRASS's 100-point layout.

    GRASS-plane only accepts the fixed layout of cst_airfoil_codec: 50 upper
    points from the trailing edge towards the leading edge, the shared leading
    edge at index 50, and 49 lower points back to the trailing edge.  Airfoil
    .dat files carry an arbitrary number of points (e.g. 52 for naca63a210), so
    both surfaces are resampled on a cosine x-grid and the trailing edge gap is
    closed by removing the mean trailing-edge offset.
    """
    import torch
    cst, _ = grass_modules()
    path = f"{GRASS}/foildata/coord_seligFmt/{name}.dat"
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue                    # title lines such as "NACA 63A210"
    raw = np.asarray(rows, float)
    le = int(np.argmin(raw[:, 0]))
    upper = raw[:le + 1][::-1]              # leading edge -> trailing edge
    lower = raw[le:]                        # leading edge -> trailing edge
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    def surface(points, x):
        x = np.clip(x, points[0, 0], points[-1, 0])
        return np.interp(x, points[:, 0], points[:, 1])

    xu = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, 50)))   # 1 -> 0
    xu[-1] = 5e-5                    # keep the shared leading edge at index 50
    xl = xu[1:][::-1]                # 0 -> 1, 49 points
    xl[-1] = 1.0                     # lower trailing-edge x must be exactly 1
    te = 0.5 * (surface(upper, 1.0) + surface(lower, 1.0))
    yu = surface(upper, xu) - te * xu
    yl = surface(lower, xl) - te * xl
    pts = torch.as_tensor(np.concatenate([
        np.stack([xu, yu], 1), np.zeros((1, 2)), np.stack([xl, yl], 1)]),
        dtype=torch.float64)
    fitted = cst.fit_airfoil_points(pts)
    return torch.as_tensor(fitted["code"], dtype=torch.float64).reshape(1, -1)


def section_geometry(sections):
    """GRASS-plane geometry dict from an ordered (root .. tip) section list.

    Each section: y, chord, le_x (default 0), z (default 0), twist_deg (default 0),
    code = (1, n) CST code or an airfoil name resolved in GRASS-plane foildata.
    """
    import torch
    cst, _ = grass_modules()
    xyz, chord, twist, codes = [], [], [], []
    for s in sections:
        xyz.append([float(s.get("le_x", 0.0)), float(s["y"]), float(s.get("z", 0.0))])
        chord.append(float(s["chord"]))
        twist.append(np.radians(float(s.get("twist_deg", 0.0))))
        code = s["code"]
        if isinstance(code, str):
            code = dat_code(code)
        codes.append(torch.as_tensor(code, dtype=torch.float64).reshape(-1))
    xyz = torch.as_tensor(np.asarray(xyz), dtype=torch.float64)
    xyz[:, 0] = xyz[:, 0] - xyz[0, 0]          # GRASS requires the root LE at x = 0
    return {"leading_edge_xyz": xyz,
            "chord": torch.as_tensor(chord, dtype=torch.float64),
            "twist_radians": torch.as_tensor(twist, dtype=torch.float64),
            "cst": torch.stack(codes)}


def _mirror(half_vertices, half_faces):
    """Mirror a right half surface onto y<0, sharing the seam vertices."""
    v = np.asarray(half_vertices, float)
    f = np.asarray(half_faces, int)
    seam = np.abs(v[:, 1]) <= 1e-12
    other = np.flatnonzero(~seam)
    index = np.arange(len(v))
    index[other] = len(v) + np.arange(len(other))
    full = np.concatenate([v, v[other] * np.array([1., -1., 1.])])
    origin = np.concatenate([np.arange(len(v)), other])
    return full, np.concatenate([f, index[f][:, ::-1]]), origin


def build_wing(sections, *, span_panels=3, chord_panels=8, wake_panels=3,
               wake_length=5.0, name="grass", reference_point=(0., 0., 0.)):
    """GRASS-plane thick wing -> panel.p1_lifting_g1.Wing (full span)."""
    from panel.p1_lifting_g1 import Wing
    _, g3d = grass_modules()

    geometry = section_geometry(sections)
    n = len(sections)
    topo = g3d.thick_wing_topology(n, span_panels, chord_panels)
    half = g3d.thick_wing_half_mesh(geometry, (0., 0., 0.), topo)
    half_vertices = np.asarray(half["vertices"].detach().cpu(), float)
    half_faces = np.asarray(half["faces"].detach().cpu(), int)

    full, faces, origin = _mirror(half_vertices, half_faces)
    perimeter = 2 * chord_panels
    rows = (n - 1) * span_panels + 1
    tip_node = rows * perimeter

    seam = np.abs(half_vertices[:, 1]) <= 1e-12
    other = np.flatnonzero(~seam)
    mirror_index = np.arange(len(half_vertices))
    mirror_index[other] = len(half_vertices) + np.arange(len(other))
    te_half = np.array([i * perimeter + chord_panels for i in range(rows)])
    left = [int(mirror_index[te_half[i]]) for i in range(rows - 1, 0, -1)]
    right = [int(te_half[i]) for i in range(rows)]
    stations = np.array(left + right, int)
    station_y = full[stations, 1]
    if not np.all(np.diff(station_y) > 1e-12):
        raise ValueError(name + ": TE stations are not strictly increasing in y")

    ring = origin[faces[:, 0]] % perimeter
    cap = origin[faces[:, 0]] == tip_node
    bank = np.where(cap, 0, np.where(ring < chord_panels, 1, -1))

    # Independent lower trace at every interior TE station.
    interior = np.arange(1, len(stations) - 1)
    dup = {int(stations[k]): len(full) + s for s, k in enumerate(interior)}
    dof = faces.copy()
    lower = bank == -1
    for node, d in dup.items():
        hit = lower & (faces == node).any(axis=1)
        dof[hit] = np.where(faces[hit] == node, d, dof[hit])
    te_down = stations.copy()
    for node, d in dup.items():
        te_down[np.flatnonzero(stations == node)] = d

    wake = []
    for k in range(wake_panels + 1):
        seg = full[stations].copy()
        seg[:, 0] += wake_length * (k / wake_panels)
        wake.append(seg)
    wake_xyz = np.concatenate(wake, 0)
    stride = len(stations)
    wake_faces = []
    for k in range(wake_panels):
        for j in range(stride - 1):
            a, b = k * stride + j, (k + 1) * stride + j
            c, d = (k + 1) * stride + j + 1, k * stride + j + 1
            wake_faces.extend([(a, b, c), (a, c, d)])
    gamma_hat = np.zeros((len(wake_xyz), len(interior)))
    for k in range(wake_panels + 1):
        for s, j in enumerate(interior):
            gamma_hat[k * stride + j, s] = 1.0

    chords = np.array([float(s["chord"]) for s in sections])
    ys = np.array([float(s["y"]) for s in sections])
    planform = float(np.trapezoid(chords, ys)) * 2.0
    return Wing(full, faces, bank, stations, stations.copy(), te_down,
                np.stack([stations[interior], te_down[interior]], 1),
                station_y[interior], dof, len(full) + len(interior), wake_xyz,
                np.asarray(wake_faces, int), station_y, gamma_hat,
                float(chords[0]), float(2 * ys[-1]), name, planform,
                np.asarray(reference_point, float))
