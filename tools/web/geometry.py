"""Geometry construction: airfoil sections, display mesh, OCC STEP solid."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from . import params as params_module


def _camber_line(x, camber, position):
    """NACA-4 camber line with its maximum at `position` of the chord."""
    if camber <= 0.0 or not (0.0 < position < 1.0):
        return 0.0
    p = position
    if x <= p:
        return camber / (p * p) * (2.0 * p * x - x * x)
    q = 1.0 - p
    return camber / (q * q) * (1.0 - 2.0 * p + 2.0 * p * x - x * x)


def _half_thickness(x, tc):
    """NACA-00XX half thickness (closed trailing edge)."""
    return 5.0 * tc * (0.2969 * math.sqrt(max(x, 0.0)) - 0.126 * x
                       - 0.3516 * x * x + 0.2843 * x ** 3 - 0.1036 * x ** 4)


def _placement(station):
    """Rotate a section-frame point about the trailing edge into body axes.

    Twist is applied in the section frame, then the section is rolled about the
    x axis by the dihedral angle through its own TE: the thickness direction
    becomes `(0, -sin(d), cos(d))` and the TE sits at `(te_x, y, |y| tan(d))`.
    Rolling (rather than merely shifting z) keeps the section a rigid airfoil,
    which is what a dihedral wing is.
    """
    c, s = math.cos(station["twist"]), math.sin(station["twist"])
    chord = station["chord"]
    te = station["te_x"]
    te_z = float(station.get("te_z", 0.0))
    dihedral = math.atan(float(station.get("dihedral", 0.0)))
    cd, sd = math.cos(dihedral), math.sin(dihedral)
    y = float(station["y"])

    def place(x, z):
        xc = (x - 1.0) * chord   # x measured from the trailing edge
        zc = z * chord
        x_off = xc * c + zc * s
        z_off = -xc * s + zc * c
        return (te + x_off, y - z_off * sd, te_z + z_off * cd)

    return place


def section_curves(station, count=41):
    """Upper and lower surface polylines of one section in body axes.

    `upper` runs LE->TE and `lower` runs TE->LE; both end exactly on the
    trailing edge at (station["te_x"], station["te_z"]) in the section plane, so
    the declared TE polyline is reproduced to machine precision in 3-D (dihedral
    included).
    """
    n = int(count)
    chord = station["chord"]
    tc = station["tc"]
    camber = station["camber"]
    position = station["position"]
    x = [0.5 * (1.0 - math.cos(math.pi * i / (n - 1.0))) for i in range(n)]
    zc = [_camber_line(value, camber, position) for value in x]
    t = [_half_thickness(value, tc) for value in x]
    place = _placement(station)
    upper = [place(x[i], zc[i] + t[i]) for i in range(n)]
    lower = [place(x[i], zc[i] - t[i]) for i in range(n - 1, -1, -1)]
    return upper, lower


def section_loop(station, count=41):
    """Closed airfoil loop in body axes (upper LE->TE then lower TE->LE)."""
    upper, lower = section_curves(station, count)
    return upper + lower[1:-1]


def section_xyz(station, count=41):
    """3D section loop at one span station."""
    return list(section_loop(station, count))


def grass_curves(p, count=41):
    """GRASS physical section frames, mirrored for the full-span ruled loft."""
    import torch
    grass = Path(__file__).resolve().parents[3] / "GRASS-plane"
    if str(grass) not in sys.path:
        sys.path.insert(0, str(grass))
    from cst_airfoil_codec import cst_surfaces_at_x
    from wing_geometry_encoder import wing_section_points

    data = p["grass_sections"]
    g = {key: torch.tensor(data[key], dtype=torch.float64) for key in
         ("leading_edge_xyz", "chord", "twist_radians", "cst")}
    xi = (1 - torch.cos(torch.linspace(0, torch.pi, count, dtype=torch.float64))) / 2
    upper, lower = cst_surfaces_at_x(g["cst"], xi)
    if bool((upper[:, 1:-1] <= lower[:, 1:-1]).any()):
        raise ValueError("CST 上表面必须位于下表面之上")
    chord = g["chord"][:, None]
    place = torch.zeros(3, dtype=torch.float64)
    up = wing_section_points(g, chord * xi, chord * upper, place).detach().numpy()
    lo = wing_section_points(g, chord * xi, chord * lower, place).detach().numpy()
    zero = torch.zeros((len(chord), 1), dtype=torch.float64)
    up_control = wing_section_points(g, zero, chord * upper[:, 1:2], place)[:, 0].detach().numpy()
    lo_control = wing_section_points(g, zero, chord * lower[:, 1:2], place)[:, 0].detach().numpy()
    positive = [dict(upper=up[i].tolist(), lower=lo[i][::-1].tolist(),
                     upper_control=up_control[i].tolist(), lower_control=lo_control[i].tolist())
                for i in range(len(chord))]
    negative = []
    for section in reversed(positive[1:]):
        negative.append({key: ([[v[0], -v[1], v[2]] for v in value]
                               if key in ("upper", "lower") else [value[0], -value[1], value[2]])
                         for key, value in section.items()})
    return negative + positive


def section_data(p, count=41):
    if "grass_sections" in p:
        return grass_curves(p, count)
    data = []
    for station in params_module.full_stations(p):
        upper, lower = section_curves(station, count)
        x1 = .5 * (1. - math.cos(math.pi / (count - 1.)))
        t1 = _half_thickness(x1, station["tc"])
        place = _placement(station)
        data.append(dict(upper=upper, lower=lower, upper_control=place(0., t1),
                         lower_control=place(0., -t1)))
    return data


def trailing_edge_points(p):
    if "grass_sections" in p:
        points = [s["upper"][-1] for s in section_data(p)]
        if any(points[i + 1][1] <= points[i][1] for i in range(len(points) - 1)):
            raise ValueError("GRASS 后缘在机体 y 方向必须严格递增")
        return points
    return params_module.trailing_edge_points(p)


def display_mesh(p, sections=None, count=41):
    """Triangulated full-span surface for the viewer (no CAD kernel involved)."""
    st = section_data(p, count)
    n_sections = int(sections or len(st))
    if n_sections < len(st):
        step = (len(st) - 1) / (n_sections - 1.0)
        st = [st[int(round(i * step))] for i in range(n_sections)]
    loops = [s["upper"] + s["lower"][1:-1] for s in st]
    n_pts = len(loops[0])
    vertices = []
    for loop in loops:
        vertices.extend(loop)
    faces = []
    for i in range(len(loops) - 1):
        a = i * n_pts
        b = (i + 1) * n_pts
        for j in range(n_pts):
            k = (j + 1) % n_pts
            faces.append([a + j, a + k, b + k])
            faces.append([a + j, b + k, b + j])
    # Closed tip caps: fan around each end section centroid.
    for idx, base in ((0, 0), (len(loops) - 1, (len(loops) - 1) * n_pts)):
        cx = sum(v[0] for v in loops[idx]) / n_pts
        cy = sum(v[1] for v in loops[idx]) / n_pts
        cz = sum(v[2] for v in loops[idx]) / n_pts
        center = len(vertices)
        vertices.append((cx, cy, cz))
        for j in range(n_pts):
            k = (j + 1) % n_pts
            if idx == 0:
                faces.append([center, base + k, base + j])
            else:
                faces.append([center, base + j, base + k])
    return dict(vertices=[[float(c) for c in v] for v in vertices], faces=faces,
                sections=len(loops), points_per_section=n_pts)


def write_trailing_edge(path, p):
    """TE declaration consumed by `panel.cad_half`."""
    ref = params_module.reference(p)
    data = dict(points=trailing_edge_points(p),
                chord=ref["chord"], reference_area=ref["area"],
                origin=[0.0, 0.0, 0.0])
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def write_mesh_features(path, p):
    """Use the actual ruled-loft section leading-edge endpoints."""
    points = [list(s["upper"][0]) for s in section_data(p)]
    data = dict(schema="dpanel.mesh_features.v1", units="m", features=[
        dict(name="leading_edge", points=points, target_ratio=.5, radius_chords=.12)])
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def write_step(path, p, count=41):
    """Loft the sections into one closed OCC solid and write it as STEP."""
    import gmsh

    if not isinstance(count, int) or count < 4:
        raise ValueError("STEP section requires at least four points per side")
    stations = section_data(p, count)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("wing")
        occ = gmsh.model.occ
        wires = []
        for s in stations:
            # Two B-splines per section (upper, lower) instead of one line per
            # sample point: a 41-point polyline loft produced ~2000 OCC faces and
            # a 3.5-minute STEP import, while two splines give one face per span
            # segment and keep the trailing edge sharp (both splines end on it).
            upper, lower = list(s["upper"]), list(s["lower"])
            # B-spline endpoint tangents follow their first control-polygon edge.
            upper[1] = s["upper_control"]
            lower[-2] = s["lower_control"]
            upper_tags = [occ.addPoint(x, y, z, tag=-1, meshSize=0.0) for x, y, z in upper]
            lower_tags = [occ.addPoint(x, y, z, tag=-1, meshSize=0.0) for x, y, z in lower]
            wires.append(occ.addCurveLoop([occ.addBSpline(upper_tags),
                                           occ.addBSpline(lower_tags)]))
        occ.synchronize()
        # Ruled loft: the trailing-edge boundary is then the straight segment
        # between consecutive section TE points, which is exactly the polyline
        # declared in trailing_edge.json. A smooth loft would bow away from that
        # polyline by ~3e-3 and `cad_te.validate_declared_te` (tol 1e-7) rejects it.
        occ.addThruSections(wires, makeSolid=True, makeRuled=True)
        occ.synchronize()
        solids = [tag for dim, tag in gmsh.model.getEntities(3)]
        if not solids:
            raise RuntimeError("loft produced no solid")
        volume = gmsh.model.occ.getMass(3, solids[-1])
        if not volume > 0.0:
            raise RuntimeError("loft produced a non-positive volume")
        gmsh.write(str(path))
        return dict(volume=float(volume), sections=len(stations),
                    points_per_section=2 * count - 2)
    finally:
        gmsh.finalize()

def body_display_mesh(bodies, nu=32, nv=16):
    """Triangulated ellipsoid fuselages for the viewer (no CAD kernel involved)."""
    vertices = []
    faces = []
    for body in bodies:
        a = 0.5 * float(body["length"])
        r = 0.5 * float(body["diameter"])
        cx = float(body["center_x"])
        cz = float(body["center_z"])
        rows = []
        for i in range(nv + 1):
            phi = math.pi * i / nv
            x = cx - a * math.cos(phi)
            ring = r * math.sin(phi)
            row = []
            for j in range(nu):
                theta = 2.0 * math.pi * j / nu
                row.append(len(vertices))
                vertices.append((x, ring * math.cos(theta), cz + ring * math.sin(theta)))
            rows.append(row)
        for i in range(nv):
            for j in range(nu):
                k = (j + 1) % nu
                a0, a1 = rows[i][j], rows[i][k]
                b0, b1 = rows[i + 1][j], rows[i + 1][k]
                faces.append([a0, a1, b1])
                faces.append([a0, b1, b0])
    return dict(vertices=[[float(c) for c in v] for v in vertices], faces=faces,
                bodies=len(bodies))


def write_body_step(path, bodies):
    """Write every fuselage as one OCC solid into a single STEP file."""
    import gmsh

    if not bodies:
        return None
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("body")
        occ = gmsh.model.occ
        for body in bodies:
            a = 0.5 * float(body["length"])
            cx = float(body["center_x"])
            cz = float(body["center_z"])
            tag = occ.addSphere(cx, 0.0, cz, 0.5)
            occ.dilate([(3, tag)], cx, 0.0, cz,
                       float(body["length"]), float(body["diameter"]), float(body["diameter"]))
        occ.synchronize()
        solids = [tag for dim, tag in gmsh.model.getEntities(3)]
        if len(solids) != len(bodies):
            raise RuntimeError("机身得到 %d 个实体，期望 %d 个" % (len(solids), len(bodies)))
        gmsh.write(str(path))
        return dict(solids=len(solids))
    finally:
        gmsh.finalize()
