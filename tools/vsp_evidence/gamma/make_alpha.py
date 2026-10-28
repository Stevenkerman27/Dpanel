"""Rotate the saved freestream to a fixed angle of attack, keeping every other byte of the half-surface.

The gamma refinement experiment must run at the operating point (5 deg), but the
saved surfaces bake in the 0 deg freestream. Only the direction rotates; speed,
mesh, wake, stations and TE declarations are untouched.
"""
import argparse, json, math, pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--alpha", type=float, default=5.0)
    a = p.parse_args()
    fixture = json.loads(a.surface.read_text(encoding="utf-8"))
    old = fixture["freestream"]
    speed = math.sqrt(sum(float(v) ** 2 for v in old))
    ang = math.radians(a.alpha)
    fixture["freestream"] = [speed * math.cos(ang), 0.0, speed * math.sin(ang)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"speed": round(speed, 6), "alpha": a.alpha,
                      "freestream": [round(v, 6) for v in fixture["freestream"]],
                      "faces": len(fixture["faces"])}))


if __name__ == "__main__":
    main()
