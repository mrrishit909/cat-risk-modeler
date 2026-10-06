"""Held-out evaluation: coasts the demo never uses (seeds 21, 31, 41). For each, the model is built the way the demo builds
it (climate from forty years of best tracks, damage curve from fifteen years of claims, geocoded locations) and compared with
the simulation truth, one ingredient at a time, and with the burn cost of the forty observed storms.

    python -m ct.evaluate > docs/evaluation.md
"""
import time

import numpy as np

from . import engine as E, world as W

SEEDS = (21, 31, 41)
EVENTS = 30000


def metrics(b, climate, vuln, seed, coords="geocoded"):
    ev, loss = E.run_catalogue(b, climate, vuln, EVENTS, seed, coords=coords)
    s, _ = E.summarise(ev, loss)
    return {"aal": s["aal_gross"], "pml100": s["oep_gross"][100], "pml250": s["oep_gross"][250], "ci250": s["ci"][250]}


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}%"


def m(v):
    return f"${v / 1e6:,.1f}M"


def main():
    t0 = time.time()
    out = ["# Evaluation", "", f"Three coasts the demo never uses (seeds {', '.join(map(str, SEEDS))}), {EVENTS:,} events per catalogue. Each row changes one",
           "ingredient from the simulation truth (true climate, true damage curves, true coordinates) towards the model the demo builds.",
           "Reproduce with `python -m ct.evaluate`.", ""]
    rows, fits = [], []
    for seed in SEEDS:
        b = W.book(seed)
        tb = {**b, "gx": b["x"], "gy": b["y"]}
        cal = E.calibrate(seed)
        truth = metrics(tb, W.TRUE_CLIMATE, W.TRUE_VULN, seed + 7)
        steps = [("Truth", truth),
                 ("+ geocoded locations", metrics(b, W.TRUE_CLIMATE, W.TRUE_VULN, seed + 7)),
                 ("+ damage curve fitted to claims", metrics(b, W.TRUE_CLIMATE, cal["vuln"], seed + 7)),
                 ("+ climate fitted to 40 years (the model)", metrics(b, cal["climate"], cal["vuln"], seed + 7)),
                 ("Model with the naive damage curve", metrics(b, cal["climate"], cal["vuln_naive"], seed + 7))]
        bc = E.burn_cost(b, seed, cal["vuln"])
        rows.append((seed, steps, bc, truth))
        st_fit = E.stress(b, cal["vuln"], W.worst_town_x(b)[1], 90, 22, 5, 0, draws=60)["gross_mean"]
        st_true = E.stress(tb, W.TRUE_VULN, W.worst_town_x(b)[1], 90, 22, 5, 0, draws=60)["gross_mean"]
        fits.append((seed, cal, st_fit, st_true))
    out += ["## Average annual loss and probable maximum loss, against the truth", "",
            "| Coast | Step | AAL | vs truth | 100-year PML | vs truth | 250-year PML | vs truth |", "|---|---|---|---|---|---|---|---|"]
    for seed, steps, bc, truth in rows:
        for name, r in steps:
            out.append(f"| {seed} | {name} | {m(r['aal'])} | {pct(r['aal'], truth['aal']) if name != 'Truth' else ''} | {m(r['pml100'])} | {pct(r['pml100'], truth['pml100']) if name != 'Truth' else ''} | {m(r['pml250'])} | {pct(r['pml250'], truth['pml250']) if name != 'Truth' else ''} |")
        out.append(f"| {seed} | Burn cost: the 40 observed storms on today's book | {m(bc['aal_gross'])} | {pct(bc['aal_gross'], truth['aal'])} | worst year {m(bc['worst_year'])} | | | |")
    out += ["", "## Sampling error of the catalogue", "", "90% interval of the 250-year PML from resampling whole years of the truth catalogue:", ""]
    for seed, steps, bc, truth in rows:
        lo, hi = truth["ci250"]
        out.append(f"- coast {seed}: {m(lo)} to {m(hi)} ({100 * (hi - lo) / 2 / truth['pml250']:.1f}% either side) at {EVENTS:,} events")
    out += ["", "## What forty years and fifteen years of claims recover", "",
            "| Coast | Storms seen | Rate (true 0.45) | P(cat 4+) per landfall (true 0.020) | Reported claims / policies in footprints | Wood frame damage at 75 m/s (true 0.48) | Naive fit at 45 m/s (true 0.018) | Cat 4 at the largest town: model / truth |",
            "|---|---|---|---|---|---|---|---|"]
    for seed, cal, st_fit, st_true in fits:
        c, v = cal["climate"], cal["vuln"]
        out.append(f"| {seed} | {cal['storms']} | {c['rate']:.2f} | {E.p_cat4(c):.3f} | {cal['claims']['reported']} / {cal['claims']['policies_in_footprints']} | {E.curve(v, gusts=[75])[0]:.2f} | {E.curve(cal['vuln_naive'], gusts=[45])[0]:.3f} | {m(st_fit)} / {m(st_true)} ({pct(st_fit, st_true)}) |")
    out += ["", f"Run time {time.time() - t0:.0f} s on one core."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
