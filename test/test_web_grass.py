import pytest
from pathlib import Path

from tools.web import params, pipeline

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "GRASS-plane" / "cst_airfoil_codec.py").is_file(),
    reason="GRASS-plane CST codec is not installed alongside this repository")


def test_grass_web_job_uses_same_mesh_for_solve_and_gradient(tmp_path):
    job = pipeline.Job(tmp_path)
    pipeline.build(job, {"params": {"grass_sections": params.GRASS_DEFAULTS}})
    pipeline.run_mesh(job, dict(size=1.1, alpha=4))
    assert job.state == "meshed", job.error
    assert job.mesh_info["mesh_type"] == "grass_fixed_topology"
    # Both paths must follow the production closure, and both must be able to name a closure
    # explicitly. The forward job used to pin kutta="pressure" in tools/web/solve_runner.py
    # while the gradient job followed panel/p1_lifting_g2.PRODUCTION_KUTTA; with the default
    # set to potential_jump that made CL differ by 1.5e-4 (0.286075 forward vs 0.286226
    # gradient, measured 2026-11-10) and it looked like two pj implementations disagreed.
    # They never did: the forward number was the pressure closure.
    lift = {}
    for kutta in ("production", "potential_jump"):
        pipeline.run_job(job, dict(size=1.1, alpha=4, wake="fixed",
                                   wall_formulation="galerkin", ncpu=1, kutta=kutta))
        assert job.state == "done", job.error
        forward = job.result
        pipeline.run_grass(job, dict(alpha=4, kutta=kutta))
        assert job.state == "done", job.error
        assert job.result["CL"] == pytest.approx(forward["CL"], abs=2e-9)
        assert job.result["CD"] == pytest.approx(forward["CD"], abs=2e-9)
        lift[kutta] = forward["CL"]
    # The two closures must still be distinguishable; otherwise the loop above would only
    # prove that the kutta option is ignored.
    assert abs(lift["production"] - lift["potential_jump"]) > 1e-6
    assert job.result["gradients"]["CL"]["leading_edge_xyz"][0] == [None, None, None]
    assert job.result["gradients"]["CD"]["cst"][0][-1] is None
    assert len(job.result["gradients"]["CL"]["chord"]) == 2


def test_grass_web_job_rejects_missing_mesh(tmp_path):
    job = pipeline.Job(tmp_path)
    pipeline.run_grass(job, dict(alpha=4))
    assert job.state == "error"
    assert "网格" in job.error
