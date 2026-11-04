"""Geometry construction: airfoil sections, display mesh, OCC STEP solid."""
from __future__ import annotations

import json
import math
from pathlib import Path

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
    """Rotate a section-frame point about the trailing edge into body axes."""
    c, s = math.cos(station["twist"]), math.sin(station["twist"])
    chord = station["chord"]
    te = station["te_x"]

    def place(x, z):
        xc = (x - 1.0) * chord   # x measured from the trailing edge
        zc = z * chord
        return (te + xc * c + zc * s, -xc * s + zc * c)

    return place


def section_curves(station, count=41):
    """Upper and lower surface polylines of one section in body axes.

    `upper` runs LE->TE and `lower` runs TE->LE; both end exactly on the
    trailing edge, which sits at (station["te_x"], z=0) in every section. That
    keeps the declared trailing-edge polyline exactly in z=0 as
    `lifting3d/cad_contact.py:validate_te_polyline` demands.
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
    return [(x, station["y"], z) for x, z in section_loop(station, count)]


def display_mesh(p, sections=None, count=41):
    """Triangulated full-span surface for the viewer (no CAD kernel involved)."""
    st = params_module.full_stations(p)
    n_sections = int(sections or max(12, 2 * int(p["sections"])))
    picked = []
    if n_sections <= len(st):
        step = (len(st) - 1) / (n_sections - 1.0)
        picked = [st[int(round(i * step))] for i in range(n_sections)]
    else:
        picked = st
    loops = [section_xyz(s, count) for s in picked]
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
        cz = sum(v[2] for v in loops[idx]) / n_pts
        center = len(vertices)
        vertices.append((cx, picked[idx]["y"], cz))
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
    data = dict(points=params_module.trailing_edge_points(p),
                chord=ref["chord"], reference_area=ref["area"],
                origin=[0.0, 0.0, 0.0])
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def write_step(path, p, count=41):
    """Loft the sections into one closed OCC solid and write it as STEP."""
    import gmsh

    stations = params_module.full_stations(p)
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
            upper, lower = section_curves(s, count)
            upper_tags = [occ.addPoint(x, s["y"], z, tag=-1, meshSize=0.0) for x, z in upper]
            lower_tags = [occ.addPoint(x, s["y"], z, tag=-1, meshSize=0.0) for x, z in lower]
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
                    points_per_section=len(section_loop(stations[0], count)))
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
