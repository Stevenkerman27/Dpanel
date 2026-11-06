import json
import os

import pytest

from tools.web import pipeline, solve_runner


def test_solve_threads_reach_the_child_command_and_result(tmp_path, monkeypatch):
    job = pipeline.Job(tmp_path)
    commands = []
    ncpu = min(3, os.cpu_count() or 1)

    def spawn(job, command, log_path):
        commands.append(command)
        directory = tmp_path / "solve"
        directory.mkdir()
        (directory / "result.json").write_text('{"CL": 0.5}', encoding="utf-8")
        return 0, log_path

    monkeypatch.setattr(pipeline, "_spawn", spawn)
    assert pipeline.solve(job, tmp_path / "surface.json", "fixed", 4., 1, {},
                          ncpu=ncpu)["CL"] == 0.5
    assert commands[0][-2:] == ["--ncpu", str(ncpu)]
    assert commands[0][commands[0].index("--wall-formulation")+1] == "galerkin"

    def fixed(surface, alpha, reference, wall_formulation):
        assert wall_formulation == "galerkin"
        assert os.environ["OPENBLAS_NUM_THREADS"] == str(ncpu)
        assert os.environ["OMP_NUM_THREADS"] == str(ncpu)
        assert os.environ["MKL_NUM_THREADS"] == str(ncpu)
        return dict(CL=0.5)

    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(solve_runner, "solve_fixed", fixed)
    out = tmp_path / "runner"
    assert solve_runner.main(["--surface", "unused.json", "--out", str(out),
                              "--ncpu", str(ncpu)]) == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["ncpu"] == ncpu
    assert result["runner_seconds"] >= 0


@pytest.mark.parametrize("ncpu", [0, -1, (os.cpu_count() or 1) + 1])
def test_runner_refuses_invalid_thread_counts(tmp_path, ncpu):
    with pytest.raises(SystemExit) as error:
        solve_runner.main(["--surface", "unused.json", "--out", str(tmp_path),
                           "--ncpu", str(ncpu)])
    assert error.value.code == 2
