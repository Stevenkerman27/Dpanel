"""Consolidated ten-geometry comparison of gate, kernel and fan effects.

Reads the gated sweep and the open sweep, prints a markdown table (and can
append it to the convergence document).  Every ratio is computed from CL only
because CL is the one functional that was shown to converge in this setting.
"""
import argparse
import json
from pathlib import Path

LABELS = {"analytic": "collocation, freeze",
          "analytic_open": "collocation, no freeze",
          "8:2/16:2": "fan 8:2/16:2",
          "16:4/32:4": "fan 16:4/32:4"}


def load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table = {}
    for record in payload["records"]:
        table[(record["case"], record["config"])] = record
    return payload, table


def cl(record):
    if record is None:
        return None
    value = record.get("CL")
    if value is None or value != value or abs(value) > 10.:
        return None
    return float(value)


def rejected(record):
    if record is None:
        return None
    eligibility = record.get("eligibility") or {}
    return eligibility.get("rejected")


def ratio(a, b):
    if a is None or b in (None, 0.):
        return None
    return a / b


def fmt(value, digits=5):
    return "n/a" if value is None else ("%+." + str(digits) + "f") % value


def fmt_ratio(value, digits=2):
    return "n/a" if value is None else ("%." + str(digits) + "fx") % value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="docs/issue84/grass_fan_convergence_fixed.json")
    parser.add_argument("--open-json",
                        default="docs/issue84/grass_fan_convergence_fixed_open.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main_payload, main = load(args.json)
    _, open_table = load(args.open_json)
    cases = sorted({case for case, _ in main})
    lines = ["| geometry | CL freeze | CL no freeze | CL fan 8:2 | CL fan 16:4 | gate ratio | kernel ratio | fan ratio |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    worst = dict(gate=0., kernel=0., fan=0.)
    for case in cases:
        frozen = cl(main.get((case, "analytic")))
        unfrozen = cl(main.get((case, "analytic_open")))
        fan8 = cl(main.get((case, "8:2/16:2")))
        fan16 = cl(main.get((case, "16:4/32:4")))
        fan8_open = cl(open_table.get((case, "8:2/16:2@open")))
        gate = ratio(frozen, unfrozen)
        kernel = ratio(fan8_open, unfrozen)
        fan = ratio(fan8, fan16)
        for key, value in (("gate", gate), ("kernel", kernel), ("fan", fan)):
            if value is not None:
                worst[key] = max(worst[key], abs(value - 1.) if key != "gate" else value)
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" %
                     (case, fmt(frozen), fmt(unfrozen), fmt(fan8), fmt(fan16),
                      fmt_ratio(gate), fmt_ratio(kernel, 4), fmt_ratio(fan, 5)))
    lines.append("")
    lines.append("worst: gate %.2fx, kernel %.5fx, fan %.6fx" %
                 (worst["gate"], worst["kernel"], 1. + worst["fan"]))
    lines.append("")
    lines.append("| geometry | rejected (freeze) | rejected (fan 8:2) | rejected (fan 16:4) |")
    lines.append("| --- | --- | --- | --- |")
    for case in cases:
        lines.append("| %s | %s | %s | %s |" % (
            case, rejected(main.get((case, "analytic"))),
            rejected(main.get((case, "8:2/16:2"))), rejected(main.get((case, "16:4/32:4")))))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print("wrote " + args.out)


if __name__ == "__main__":
    main()
