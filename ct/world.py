"""The simulated coast. A property book on a straight 600 km coastline, the true hurricane climate and the true damage curves
(which the models never see), and forty years of history that the models learn from: best tracks with measurement error and,
for the last fifteen years, claims, where a loss below the deductible is never reported.

Units: km (x along the coast, y inland), m/s (1-minute sustained wind at 10 m; gusts are 1.25x), hPa, metres, USD.
"""
import functools
import math

import numpy as np
from scipy.special import ndtr

COAST_KM = 600
HISTORY_YEARS = 40
CLAIM_YEARS = 15
GUST = 1.25
DECAY_VB, DECAY_ALPHA = 13.7, 0.095            # Kaplan-DeMaria inland decay: floor (m/s) and rate per hour
CLASSES = ["wood_frame", "masonry", "reinforced_concrete", "manufactured_home"]
CLASS_MIX = [0.5, 0.3, 0.08, 0.12]
TOWNS = [("Cape Marlow", 70, 0.10), ("Saltmarsh", 150, 0.08), ("Bayport", 235, 0.16), ("Gull Harbor", 300, 0.07),
         ("Port Ellis", 395, 0.27), ("Redfish Key", 455, 0.12), ("Palmetto Bluff", 520, 0.12), ("Eastwater", 575, 0.08)]

# --- the truth: never shown to a model -------------------------------------------------------------------------------
TRUE_CLIMATE = {"rate": 0.45, "dp_shape": 1.6, "dp_scale": 36.0, "rmax_a": 3.75, "rmax_b": -0.0095, "rmax_sd": 0.33,
                "vt_mu": math.log(5.5), "vt_sd": 0.35, "heading_mu": 0.0, "heading_sd": 25.0, "x_tilt": 0.5}
TRUE_VULN = {"v50": {"wood_frame": 76.0, "masonry": 86.0, "reinforced_concrete": 104.0, "manufactured_home": 58.0}, "beta": 0.25}
PHI = 6.0                 # secondary uncertainty: realised damage ratio ~ Beta(mean m, concentration PHI)
DP_MIN, DP_MAX = 8.0, 120.0
REINSURANCE = {"retention": 40e6, "limit": 160e6}        # per-occurrence catastrophe excess of loss


def v0(dp):
    return 6.3 * np.sqrt(dp)


def category(v):
    """Saffir-Simpson from 1-minute sustained wind in m/s."""
    return 0 if v < 33 else 1 if v < 43 else 2 if v < 50 else 3 if v < 58 else 4 if v < 70 else 5


@functools.lru_cache(maxsize=8)
def book(seed=11, n=10000):
    """The property book. True coordinates, plus the coordinates the geocoder can give: 84% rooftop, the rest only a
    postal code, which is placed at the centroid of a 10 km x 6 km postal zone."""
    r = np.random.default_rng(seed)
    w = np.array([t[2] for t in TOWNS])
    town = r.choice(len(TOWNS), n, p=w / w.sum())
    x = np.clip(np.array([TOWNS[k][1] for k in town]) + r.normal(0, 14, n), 1, COAST_KM - 1)
    y = np.minimum(r.exponential(5.0, n) + 0.05, 60) + np.where(r.random(n) < 0.12, r.uniform(5, 45, n), 0)
    y = np.minimum(y, 60)
    elev = 0.8 + 0.28 * y + r.gamma(2.0, 0.6, n)
    cls = r.choice(4, n, p=CLASS_MIX)
    cls = np.where((y > 15) & (r.random(n) < 0.15), 3, cls)                  # more manufactured homes inland
    commercial = r.random(n) < 0.14
    cls = np.where(commercial & (cls == 3), 1, cls)
    year = r.integers(1950, 2026, n)
    hip = r.random(n) < 0.4
    stories = np.where(cls == 2, r.integers(2, 13, n), r.integers(1, 3, n))
    bldg = np.where(commercial, r.lognormal(math.log(2.2e6), 0.7, n), r.lognormal(math.log(3.4e5), 0.45, n))
    bldg = np.where(cls == 3, bldg * 0.3, bldg)
    contents = bldg * np.where(commercial, 0.8, 0.5)
    bi = np.where(commercial, bldg * r.uniform(0.15, 0.4, n), bldg * 0.1)      # business interruption or living expenses, a year
    ded_pct = np.where(commercial, 0.0, r.choice([0.02, 0.05], n, p=[0.6, 0.4]))
    ded_flat = np.where(commercial, r.choice([25e3, 50e3, 100e3, 250e3], n), 0.0)
    surge_covered = commercial & (r.random(n) < 0.6)                          # homeowners' policies exclude storm surge
    rooftop = r.random(n) < 0.84
    gx = np.where(rooftop, x, (np.floor(x / 10) + 0.5) * 10)
    gy = np.where(rooftop, y, (np.floor(y / 6) + 0.5) * 6)
    order = np.argsort(gx)
    cols = dict(town=town, x=x, y=y, gx=gx, gy=gy, rooftop=rooftop, elev=elev, cls=cls, commercial=commercial, year=year, hip=hip,
                stories=stories, bldg=bldg, contents=contents, bi=bi, ded_pct=ded_pct, ded_flat=ded_flat, surge_covered=surge_covered)
    return {k: v[order] for k, v in cols.items()}


def tiv(b):
    return b["bldg"] + b["contents"] + b["bi"]


def sample_events(climate, n_years=None, n_events=None, rng=None):
    """Poisson years of landfalls from a climate. -> dict of arrays, one per event, with its year."""
    rng = rng or np.random.default_rng(0)
    if n_events is not None:
        n_years = int(n_events / climate["rate"] * 1.05) + 10
    counts = rng.poisson(climate["rate"], n_years)
    if n_events is not None:
        keep = np.searchsorted(np.cumsum(counts), n_events)
        counts = counts[:keep + 1]
        counts[-1] -= counts.sum() - n_events
        n_years = len(counts)
    m = int(counts.sum())
    yr = np.repeat(np.arange(n_years), counts)
    u = rng.random(m)
    lo = math.exp(-(DP_MIN / climate["dp_scale"]) ** climate["dp_shape"])
    hi = math.exp(-(DP_MAX / climate["dp_scale"]) ** climate["dp_shape"])
    dp = climate["dp_scale"] * (-np.log(lo - u * (lo - hi))) ** (1 / climate["dp_shape"])       # truncated Weibull
    rmax = np.exp(climate["rmax_a"] + climate["rmax_b"] * dp + rng.normal(0, climate["rmax_sd"], m))
    vt = np.exp(rng.normal(climate["vt_mu"], climate["vt_sd"], m))
    heading = rng.normal(climate["heading_mu"], climate["heading_sd"], m)
    x0 = landfall_x(climate["x_tilt"], rng.random(m))
    return {"year": yr, "x0": x0, "dp": dp, "rmax": np.clip(rmax, 8, 90), "vt": np.clip(vt, 1.5, 15), "heading": np.clip(heading, -70, 70), "n_years": n_years}


def landfall_x(tilt, u):
    """Inverse CDF of a linear density on [0, 1]: f(z) = 1 + tilt (2z - 1)."""
    if abs(tilt) < 1e-9:
        return u * COAST_KM
    a = tilt
    z = (-(1 - a) + np.sqrt((1 - a) ** 2 + 4 * a * u)) / (2 * a)
    return z * COAST_KM


def footprint(x0, dp, rmax, vt, heading, X, Y, elev):
    """Peak gust (m/s) and surge depth (m) at points X, Y. Storm parameters are scalars or arrays aligned with the points."""
    th = np.radians(heading)
    ux, uy = np.sin(th), np.cos(th)
    dx, dy = X - x0, Y
    s = dx * ux + dy * uy                       # km along the track past landfall
    d = dx * uy - dy * ux                       # km off the track, + = right of the motion
    vland = v0(dp)
    t = np.maximum(s, 0) / (vt * 3.6)
    vc = np.where(vland > DECAY_VB, DECAY_VB + (vland - DECAY_VB) * np.exp(-DECAY_ALPHA * t), vland)
    r = np.abs(d) + 0.5
    prof = np.where(r < rmax, 1.0, (rmax / r) ** 0.55)      # inside the radius of maximum wind the eyewall passes over: the peak is the eyewall's
    v = (vc + 0.5 * vt * np.clip(d / rmax, -1, 1)) * prof
    gust = (GUST * 0.85) * v + (GUST * 0.15) * np.exp(-Y / 2.0) * v
    peak = 0.06 * dp * np.where(d > 0, np.exp(-((d - rmax) / (2 * rmax)) ** 2), 0.35 * np.exp(-((d + rmax) / (2 * rmax)) ** 2))
    depth = np.maximum(peak * np.exp(-Y / 1.2) - elev, 0)
    return gust, depth


def cutoff_km(dp, rmax, vt):
    """Distance off the track beyond which the gust cannot pass 26 m/s, where every damage curve is below 1e-4."""
    vmax = GUST * (v0(dp) + 0.5 * vt)
    return rmax * np.maximum(vmax / 26.0, 1.0) ** (1 / 0.55) + 2


def x_window(x0, heading, cut, y_max=60.0):
    """The x range that can lie within `cut` km of a straight track crossing the coast at x0 (any y from 0 to y_max)."""
    th = np.radians(heading)
    drift = y_max * np.tan(th)
    half = cut / np.cos(th)
    return x0 + np.minimum(0.0, drift) - half, x0 + np.maximum(0.0, drift) + half


def v50_of(cls, year, hip, vuln):
    return np.array([vuln["v50"][c] for c in CLASSES])[cls] * np.where(year >= 2002, 1.08, 1.0) * np.where(hip, 1.05, 1.0)


def mean_damage(gust, depth, cls, year, hip, vuln, v50=None):
    """Mean damage ratio of the building, and the share of it that is surge."""
    v50 = v50_of(cls, year, hip, vuln) if v50 is None else v50
    wind = ndtr(np.log(np.maximum(gust, 1e-3) / v50) / vuln["beta"])
    flood = (1 - np.exp(-depth / 1.6)) * 0.85 * np.where(cls == 2, 0.6, 1.0)
    total = 1 - (1 - wind) * (1 - flood)
    return total, np.where(total > 0, flood * (1 - wind) / np.maximum(total, 1e-12), 0)


def financial(b, idx, ratio, surge_share, T=None):
    """Ground-up and gross (after deductible and limit) loss per location for realised building damage ratios."""
    covered = np.where(b["surge_covered"][idx], 1.0, 1 - surge_share)
    ru = np.clip(ratio * covered, 0, 1)
    gu = b["bldg"][idx] * ru + b["contents"][idx] * 0.6 * ru ** 1.3 + b["bi"][idx] * np.minimum(1.0, 2.0 * ru)
    ded = b["ded_pct"][idx] * b["bldg"][idx] + b["ded_flat"][idx]
    gross = np.minimum(np.maximum(gu - ded, 0), (tiv(b) if T is None else T)[idx])
    return gu, gross


def realise(m, rng):
    """Draw realised damage ratios around the mean (secondary uncertainty)."""
    out = np.zeros_like(m)
    k = m > 1e-5
    mm = np.clip(m[k], 1e-6, 1 - 1e-6)
    out[k] = rng.beta(mm * PHI, (1 - mm) * PHI)
    return out


def event_losses(b, ev, vuln, coords="geocoded", rng=None, keep_location=None, chunk=500):
    """Gross loss per event for the whole book, events in vectorised chunks of (event, location) pairs.
    With keep_location, also the per-location detail for that event index."""
    rng = rng or np.random.default_rng(1)
    X, Y = (b["gx"], b["gy"]) if coords == "geocoded" else (b["x"], b["y"])
    order = np.argsort(X, kind="stable")
    Xs = X[order]
    X32, Y32, E32 = X.astype(np.float32), Y.astype(np.float32), b["elev"].astype(np.float32)
    T = tiv(b)
    v50 = v50_of(b["cls"], b["year"], b["hip"], vuln)
    g_min = v50 * np.exp(-4.26 * vuln["beta"])      # ndtr(-4.26) = 1e-5
    n = len(ev["x0"])
    gross, gu, surge = np.zeros(n), np.zeros(n), np.zeros(n)
    by_location = np.zeros(len(X))
    detail = None
    for c0 in range(0, n, chunk):
        e = np.arange(c0, min(n, c0 + chunk))
        cut = cutoff_km(ev["dp"][e], ev["rmax"][e], ev["vt"][e])
        a, z = x_window(ev["x0"][e], ev["heading"][e], cut)
        lo, hi = np.searchsorted(Xs, a), np.searchsorted(Xs, z)
        cnt = hi - lo
        if cnt.sum() == 0:
            continue
        pe = np.repeat(np.arange(len(e)), cnt)                                  # pair -> event within the chunk
        pl = order[np.repeat(lo - np.cumsum(cnt) + cnt, cnt) + np.arange(cnt.sum())]   # pair -> location
        th = np.radians(ev["heading"][e])[pe]
        keep = np.abs((X[pl] - ev["x0"][e][pe]) * np.cos(th) - Y[pl] * np.sin(th)) < cut[pe]
        pe, pl = pe[keep], pl[keep]
        k = e[pe]
        f = lambda a: a.astype(np.float32)     # noqa: E731  (pairs are the volume: single precision halves the time)
        g, dpt = footprint(f(ev["x0"][k]), f(ev["dp"][k]), f(ev["rmax"][k]), f(ev["vt"][k]), f(ev["heading"][k]), X32[pl], Y32[pl], E32[pl])
        g, dpt = g.astype(np.float64), dpt.astype(np.float64)
        live = (g > g_min[pl]) | (dpt > 0)          # below g_min the mean wind damage is under 1e-5: nothing to draw
        pe, pl, g, dpt = pe[live], pl[live], g[live], dpt[live]
        m, ss = mean_damage(g, dpt, b["cls"][pl], b["year"][pl], b["hip"][pl], vuln, v50[pl])
        ratio = realise(m, rng)
        u, gr = financial(b, pl, ratio, ss, T)
        gross[e] = np.bincount(pe, gr, minlength=len(e))
        gu[e] = np.bincount(pe, u, minlength=len(e))
        surge[e] = np.bincount(pe, gr * np.where(b["surge_covered"][pl], ss, 0), minlength=len(e))
        by_location += np.bincount(pl, gr, minlength=len(X))
        if keep_location is not None and c0 <= keep_location < c0 + len(e):
            j = pe == keep_location - c0
            detail = {"idx": pl[j], "gust": g[j], "depth": dpt[j], "mean": m[j], "ratio": ratio[j], "gross": gr[j], "surge_share": ss[j]}
    return {"gross": gross, "ground_up": gu, "surge": surge, "detail": detail, "by_location": by_location}


def net_of_reinsurance(gross, ri=REINSURANCE):
    recovery = np.minimum(np.maximum(gross - ri["retention"], 0), ri["limit"])
    return gross - recovery


# --- history ---------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def history(seed=11):
    """Forty years of landfalls from the true climate, recorded with measurement error, and fifteen years of claims on the
    book of the time (a sample of today's locations): a claim exists only when the loss beats the deductible."""
    r = np.random.default_rng(seed + 1000)
    ev = sample_events(TRUE_CLIMATE, n_years=HISTORY_YEARS, rng=r)
    m = len(ev["x0"])
    observed = {"year": ev["year"] + 1986, "x0": ev["x0"] + r.normal(0, 4, m), "dp": np.clip(ev["dp"] + r.normal(0, 3, m), DP_MIN, DP_MAX),
                "rmax": ev["rmax"] * np.exp(r.normal(0, 0.15, m)), "vt": ev["vt"] * np.exp(r.normal(0, 0.1, m)), "heading": ev["heading"] + r.normal(0, 5, m)}
    b = book(seed)
    sample = np.arange(len(b["x"]))
    claims = []
    for i in range(m):
        if ev["year"][i] < HISTORY_YEARS - CLAIM_YEARS:
            continue
        g, dpt = footprint(ev["x0"][i], ev["dp"][i], ev["rmax"][i], ev["vt"][i], ev["heading"][i], b["x"][sample], b["y"][sample], b["elev"][sample])
        mw, _ = mean_damage(g, np.zeros_like(dpt), b["cls"][sample], b["year"][sample], b["hip"][sample], TRUE_VULN)   # wind claims only
        k = mw > 1e-4
        ratio = realise(mw[k], r)
        idx = sample[k]
        ded_ratio = b["ded_pct"][idx] + b["ded_flat"][idx] / b["bldg"][idx]
        for j, loc in enumerate(idx):
            claims.append((i, int(loc), float(ratio[j]) if ratio[j] > ded_ratio[j] else None, float(ded_ratio[j])))
    return {"truth": ev, "observed": observed, "claims": claims, "n_storms": m}


def worst_town_x(b):
    """The landfall point the stress test aims at: the centre of the largest insured value."""
    t = tiv(b)
    k = max(range(len(TOWNS)), key=lambda j: t[b["town"] == j].sum())
    return TOWNS[k][0], TOWNS[k][1]
