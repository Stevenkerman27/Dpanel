"""Inventory the numpy/scipy linear algebra reachable from the production G1/G2 path."""
import ast
import pathlib
import re
import sys

if len(sys.argv) > 1:
    ROOT = pathlib.Path(sys.argv[1]).resolve()
else:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / 'panel'

SEEDS = ['p1_lifting_g1.py', 'p1_lifting_g2.py', 'p1_coupled_material_wake.py']

PATTERNS = [
    ('least_squares', re.compile(r'\bleast_squares\b')),
    ('null_space', re.compile(r'\bnull_space\b')),
    ('lstsq', re.compile(r'\blstsq\b')),
    ('svd', re.compile(r'\bsvd\b(?!\w)')),
    ('qr', re.compile(r'\.qr\(|\bqr\(')),
    ('solve', re.compile(r'linalg\.solve|spsolve|splu|spsolve_triangular')),
    ('pinv', re.compile(r'\bpinv\b')),
    ('eig', re.compile(r'\beig\b|\beigh\b|\beigvals')),
    ('det/slogdet', re.compile(r'\bslogdet\b|linalg\.det')),
    ('norm', re.compile(r'linalg\.norm')),
    ('scipy-import', re.compile(r'^\s*(from|import)\s+scipy')),
    ('scipy-call', re.compile(r'^\s*(from|import)\s+scipy|\bsp\.')),
]


def imports_of(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ''
            if node.level and mod:
                out.add(mod.split('.')[0])
            elif mod.startswith('panel.'):
                out.add(mod.split('.')[1])
            elif mod == 'panel':
                for a in node.names:
                    out.add(a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith('panel.'):
                    out.add(a.name.split('.')[1])
    return out


reachable = {}
queue = list(SEEDS)
while queue:
    name = queue.pop()
    if name in reachable:
        continue
    path = PKG / name
    if not path.exists():
        continue
    reachable[name] = path
    for dep in imports_of(path):
        dep_file = dep if dep.endswith('.py') else dep + '.py'
        if (PKG / dep_file).exists():
            queue.append(dep_file)

rows = []
totals = {}
for name in sorted(reachable):
    text = reachable[name].read_text(encoding='utf-8').splitlines()
    file_hits = {}
    for i, line in enumerate(text, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for label, rx in PATTERNS:
            if rx.search(line):
                file_hits.setdefault(label, []).append((i, stripped))
                totals[label] = totals.get(label, 0) + 1
    if file_hits:
        rows.append((name, file_hits))

print('ROOT', ROOT)
print('reachable files:', len(reachable))
print('  ' + ', '.join(sorted(reachable)))
print()
for label, count in sorted(totals.items(), key=lambda kv: -kv[1]):
    print('%-14s %d' % (label, count))
print()
for name, file_hits in rows:
    print('##', name)
    seen = set()
    for label in [p[0] for p in PATTERNS]:
        for ln, txt in file_hits.get(label, []):
            if (ln, label) in seen:
                continue
            seen.add((ln, label))
            print('   %-14s %s:%d  %s' % (label, name, ln, txt[:160]))