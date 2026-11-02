import numpy as np, torch, scipy.linalg as sla
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(7)
A = rng.normal(size=(40, 12)) @ rng.normal(size=(12, 12))
b = rng.normal(size=40)
out = {}
for drv in ('gels', 'gelsy', 'gelsd'):
    try:
        sol = torch.linalg.lstsq(torch.from_numpy(A), torch.from_numpy(b), driver=drv)
        ref = np.linalg.lstsq(A, b, rcond=None)[0]
        out['lstsq_' + drv] = float(np.linalg.norm(sol.solution.numpy() - ref))
    except Exception as exc:
        out['lstsq_' + drv] = 'ERR ' + type(exc).__name__ + ': ' + str(exc)[:120]
U, S, Vh = torch.linalg.svd(torch.from_numpy(A), full_matrices=True)
S_np = np.linalg.svd(A, full_matrices=True)[1]
out['svd_spectrum_maxabs'] = float(np.max(np.abs(S.numpy() - S_np)))
out['rank_cut_1e-11'] = float(1e-11 * S_np[0])
out['sv_gap_to_cut'] = float(np.min(np.abs(S_np - 1e-11 * S_np[0])))
R = np.triu(np.linalg.qr(rng.normal(size=(20, 20)))[1])
rhs = rng.normal(size=(20, 1))
t_sol = torch.linalg.solve_triangular(torch.from_numpy(R), torch.from_numpy(rhs), upper=True)
out['solve_triangular'] = float(np.max(np.abs(t_sol.numpy() - sla.solve_triangular(R, rhs))))
Q, R2 = torch.linalg.qr(torch.from_numpy(A))
out['qr_orthonormal'] = float(np.max(np.abs((Q.T @ Q).numpy() - np.eye(Q.shape[1]))))
ns_np = sla.null_space(A, rcond=1e-11)
ns_t = torch.linalg.svd(torch.from_numpy(A), full_matrices=True)[2][S.shape[0]:].T if False else None
out['torch_version'] = torch.__version__
out['cuda'] = torch.cuda.is_available()
for k, v in out.items():
    print(k, v)