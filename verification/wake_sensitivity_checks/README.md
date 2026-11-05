# B3 frozen-evidence verification

Run from the repository root in the project myml environment, with
OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS set to 1:

```powershell
python -m tools.wake_sensitivity_verify --out runs/verification/wake_check_new/receipt.json
```

The output must not exist. The command requires the original local CAD inputs
and iteration snapshots identified by the committed experiment manifests.
Those large snapshots are not included in Git; a clone alone is insufficient.

`validated_receipt.json` is the successful run: 139 manifest-file hashes,
183 baseline snapshot hashes, frozen geometry, saved central-difference
arithmetic, E2/E3 initial wall/Gamma-space invariants and recorded E4 spectral
arithmetic were checked. The six fresh solves (iteration 0 and 1 for rect,
swept30 and elliptic) match saved CL, Gamma, Cp and wake coordinates, including
zero maximum differences in this run. Runtime was 134.28 seconds on Windows 11,
myml Python 3.13.14 / NumPy 2.4.6 / Torch 2.13.0+cu132, CPU, one thread.

This is not a full independent rerun of E1-E4 or the 60-step trajectories.
Recorded Lanczos residuals were inspected, but QR factors and spectra were not
independently reconstructed. Recorded geometry audits do not implement a full
wake/wake self-intersection test. Physical accuracy remains unaccepted.

An independent declared-TE comparison found 7/63/83 matching wake nodes, while
the original wall-coincident definition has 7/65/85. The two extra CAD contacts
are wing tips with zero Gamma basis rows. Under uniform axial flow, respectively
0/62/83 declared-TE matching nodes move. Thus the counterexample also holds on
the declared-TE subset; boundary construction for a production repair still
requires its own topology contract.

`receipt.json` and `confirmed_receipt.json` are incomplete verifier-development
attempts, not successful receipts. Their logs are preserved: the initial
verifier mistakenly equated the incomplete Kutta/TE list with every wall
contact, then included unused vertices when checking wall coincidence.
The successful verifier uses only vertices referenced by actual wall faces
and reports declared-TE and extra wall-contact sets separately. These attempts
did not modify the original experiment evidence.

The related four-file pytest run reported 30 passed, 1 skipped and 1 xfailed:
`test_wake_sensitivity_experiments.py`, `test_issue92_wake_profile.py`,
`test_analytic_batch.py`, and `test_wake_velocity_methods.py`.
The xfail reproduces the known uniform-axial-flow mapping defect.
