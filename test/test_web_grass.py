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
    pipeline.run_job(job, dict(size=1.1, alpha=4, wake="fixed",
                               wall_formulation="galerkin", ncpu=1))
    assert job.state == "done", job.error
    forward = job.result
    pipeline.run_grass(job, dict(alpha=4))
    assert job.state == "done", job.error
    assert job.result["CL"] == pytest.approx(forward["CL"], abs=2e-9)
    assert job.result["CD"] == pytest.approx(forward["CD"], abs=2e-9)
    assert job.result["gradients"]["CL"]["leading_edge_xyz"][0] == [None, None, None]
    assert job.result["gradients"]["CD"]["cst"][0][-1] is None
    assert len(job.result["gradients"]["CL"]["chord"]) == 2


def test_grass_web_job_rejects_missing_mesh(tmp_path):
    job = pipeline.Job(tmp_path)
    pipeline.run_grass(job, dict(alpha=4))
    assert job.state == "error"
    assert "网格" in job.error
