"""Measure the two-sided PV wake-velocity estimate on the real half-model CAD."""
import json, pathlib, time
import numpy as np

from panel.p1_lifting_g1 import from_dpanel_half_surface, make_wing
from panel.p1_lifting_g2 import combine_wings, solve_multi
from panel.p1_steady_free_wake import mean_wake_velocity

SURFACE = r"D:/3D/Projects/panel/runs/half_cad_wingbody_v1/surface.json"
OUT = str(pathlib.Path(__file__).resolve().parents[3] / "docs" / "issue49_vspaero" / "p1" / "free_wake_probe.json")


def build():
    main = from_dpanel_half_surface(SURFACE)
    tail = make_wing(chord_panels=4, span_panels=4, span=1.8, chord=.55, camber=0.,
                     wake_panels=6)
    return combine_wings([main, tail], origins=[(0, 0, 0), (1.6, 0, .35)],
                         names=["main_wing", "raised_tail"],
                         reference_chord=main.chord, reference_area=4.0,
                         reference_point=main.reference_point)


def main():
    model = build()
    state = solve_multi(model, 5.0, kutta="pressure")
    rows = []
    for floor in (0.0, 0.3, 0.6, 1.2):
        t0 = time.perf_counter()
        _, summary = mean_wake_velocity(model, state, probe_floor_fraction=floor)
        for name, item in summary.items():
            rows.append(dict(probe_floor_fraction=floor, group=name,
                             **{k: float(v) for k, v in item.items()}))
        print("floor %4.2f  %5.1f s" % (floor, time.perf_counter() - t0), flush=True)
    payload = dict(schema="dpanel.p1.free_wake_probe.v1",
                   source_half_surface=SURFACE, alpha_deg=5.0,
                   physical_accuracy_accepted=False,
                   note="initial (undeformed) wake field only; no relaxation",
                   rows=rows)
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(OUT).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print("%s floor %.1f slip_rms %.4f max %.4f probe_over_short %.2f aspect_p90 %.1f"
              % (row["group"], row["probe_floor_fraction"], row["normal_slip_rms"],
                 row["normal_slip_max"], row["probe_over_shortest_max"], row["aspect_p90"]))


if __name__ == "__main__":
    main()
