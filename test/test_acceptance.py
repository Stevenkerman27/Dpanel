"""The acceptance harness must stay cheap to trust: format and registry only, no solves."""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import acceptance  # noqa: E402


def _report(configs, baseline, rows, reference=False):
    return dict(
        environment=dict(commit="deadbeefcafe", dirty=False, platform="test", python="3.13",
                         numpy="2", scipy="1.18", torch="2.13", torch_threads=8,
                         generated="2026-11-10T00:00:00"),
        cases=[dict(name="demo", title="Demo", description="A demo case.", configs=configs,
                    baseline=baseline, rows=rows, seconds=1.0, notes=["a note"])])


def _row(quantity, values, unit="-", reference=None):
    row = dict(quantity=quantity, unit=unit, values=values)
    if reference is not None:
        row["reference"] = dict(label="expected", value=reference)
    return row


def test_every_registered_case_is_runnable():
    assert {"closures", "refinement", "lift_slope", "span_efficiency", "wake_material"} <= set(acceptance.CASES)
    for name, spec in acceptance.CASES.items():
        assert spec["title"] and spec["description"], name
        assert callable(spec["run"]), name


def test_list_prints_every_case():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert acceptance.main(["--list"]) == 0
    for name in acceptance.CASES:
        assert name in buffer.getvalue()


def test_markdown_reports_the_baseline_once_and_deltas_for_the_others():
    report = _report(["pressure", "potential_jump"], "pressure",
                     [_row("CL", {"pressure": 0.20, "potential_jump": 0.25})])
    markdown = acceptance.render_markdown(report)
    header = [line for line in markdown.splitlines() if line.startswith("| quantity")][0]
    assert header.count("delta potential_jump") == 1
    assert "delta pressure" not in header
    row = [line for line in markdown.splitlines() if line.startswith("| CL |")][0]
    assert "+0.05 (+25.00%)" in row


def test_markdown_omits_the_relative_error_when_the_reference_is_zero():
    report = _report(["current"], "current", [_row("drift", {"current": 0.0}, reference=0.0)])
    markdown = acceptance.render_markdown(report)
    reference_cell = [line for line in markdown.splitlines() if line.startswith("| drift |")][0]
    assert "expected 0.000000" in reference_cell
    assert "%" not in reference_cell


def test_missing_values_render_as_a_dash_instead_of_raising():
    cells, deltas = acceptance._row_cells(_row("CL", {"a": 0.2}), ["a", "b"], "a")
    assert cells == ["0.200000", "-"]
    assert deltas == [""]


def test_scientific_formatting_for_small_and_large_numbers():
    assert acceptance._format(5.35e-4) == "0.000535" or acceptance._format(5.35e-4) == "0.000535001"
    assert acceptance._format(1e-5).startswith("1e-05") or "e-05" in acceptance._format(1e-5)
    assert acceptance._format(2048.0) == "2048.000000"
