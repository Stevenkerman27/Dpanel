# Half-Mesh TE Bank Verification

The frozen input comes from the user's 2026-11-05 Web STEP geometry:
sweep 12 degrees, root/tip twist 6/-1 degrees, camber 0.02, span 4 m,
root/tip chord 1/0.5 m. The audit rebuilt that STEP at size 0.1 m with
Gmsh 4.15.2 and one mesh thread. `bank_audit.json` records the input hashes.
This is a topology/export regression, not an aerodynamic accuracy gate.

`half_mesh_input.npz` contains the pre-export conforming Gmsh arrays and
exporter arguments. The old exporter at `b12a3ff` rejects this exact frozen
mesh. The corrected exporter passes it, matches the successful CLI mesh,
and produces an unchanged dictionary for the original thick-wing control.
The original failure and the successful CLI/solve logs remain under
`runs/verification/te_bank_classification/`.

From the repository root, in the required myml environment:

```powershell
python -m tools.te_bank_verify --case verification/te_bank_classification --out runs/scratch/te_bank_replay.json
python -m pytest -q test/test_half_surface_mesh.py -p no:cacheprovider
```

To regenerate the CAD diagnostic from the local original STEP:

```powershell
python -m tools.te_bank_audit --case runs/web/20261105-172820-ad38bc --size .1 --ncpu 1 --out runs/verification/te_bank_classification/rebuild
```

Use a fresh output directory. The probe records the old height test for
comparison even when the current exporter succeeds.
