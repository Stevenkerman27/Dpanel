"""The live P1 solve chain must not call NumPy/SciPy LAPACK.

Rationale: after a torch kernel has run in the same process, the next NumPy/SciPy
LAPACK call can kill the interpreter with 0xC0000005 (docs/ISSUE92_TORCH_PORT_MAP.md
section 6f). The coupled free-wake loop calls torch kernels and then geometry
predicates, so a stray np.linalg.svd inside the loop is a latent crash, not a
style issue. This module makes that an executable assertion.

Allowed exception, with its reason inline: panel/p1_closed_body.py holds
qr/svd/solve inside its own standalone closed-body demo (run_p1_closed.py); the P1
solve imports Mesh and influence_block from it and never enters that function.
"""
import ast
import pathlib

PANEL = pathlib.Path(__file__).resolve().parents[1] / "panel"

ENTRIES = (
    "p1_lifting_g1",
    "p1_lifting_g2",
    "p1_coupled_material_wake",
    "p1_solve_torch",
    "p1_implicit_adjoint",
    "p1_steady_free_wake",
    "p1_regularized_sheet_velocity",
    "p1_analytic_sheet_velocity",
    "p1_projection_duffy",
    "p1_hybrid_area",
    "p1_adaptive_area",
    "p1_batched_influence",
)

LAPACK = {
    "svd", "svdvals", "qr", "solve", "lstsq", "eig", "eigh", "eigvals",
    "eigvalsh", "cholesky", "pinv", "matrix_rank", "slogdet", "det",
}

# path -> reason it is allowed to keep a LAPACK call
ALLOWED = {
    "p1_closed_body.py":
        "qr/svd/solve belong to the standalone closed-body demo solve(); the P1 chain "
        "imports Mesh and influence_block only and never calls solve()",
}


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _imports(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level in (0, 1) and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
    return names


def _lapack_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func)
            if name.startswith(("np.linalg.", "numpy.linalg.")):
                if name.rsplit(".", 1)[-1] in LAPACK:
                    found.append((node.lineno, name))
            elif name.startswith(("scipy.linalg.", "scipy.optimize.")):
                found.append((node.lineno, name))
    return found


def _scipy_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "scipy":
                bad.append((node.lineno, "from " + node.module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "scipy":
                    bad.append((node.lineno, "import " + alias.name))
    return bad


def _reachable(entries):
    modules = {p.stem: p for p in PANEL.glob("*.py")}
    seen, stack = set(), [e for e in entries if e in modules]
    while stack:
        stem = stack.pop()
        if stem in seen:
            continue
        seen.add(stem)
        for name in _imports(ast.parse(modules[stem].read_text(encoding="utf-8"))):
            name = name.split(".")[0]
            if name in modules and name not in seen:
                stack.append(name)
    return sorted(seen)


def test_the_live_p1_chain_calls_no_numpy_or_scipy_lapack():
    offenders = {}
    for stem in _reachable(ENTRIES):
        path = PANEL / (stem + ".py")
        if path.name in ALLOWED:
            continue
        calls = _lapack_calls(path)
        scipy = _scipy_imports(path)
        if calls or scipy:
            offenders[path.name] = calls + scipy
    assert offenders == {}, (
        "NumPy/SciPy LAPACK on the live P1 path; port it to torch.linalg or move it "
        "off the path (it can crash the process after a torch kernel): %r" % (offenders,)
    )


def test_the_allowed_exception_is_still_the_only_lapack_user():
    users = set()
    for path in PANEL.glob("*.py"):
        if _lapack_calls(path) or _scipy_imports(path):
            users.add(path.name)
    assert set(ALLOWED) <= users, "allowlisted module no longer calls LAPACK; drop it"


def test_the_scanner_actually_detects_a_lapack_call(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "import numpy as np\n"
        "def f(a):\n"
        "    return np.linalg.svd(a)[1]\n", encoding="utf-8")
    assert _lapack_calls(planted) == [(3, "np.linalg.svd")]
    assert _scipy_imports(planted) == []
