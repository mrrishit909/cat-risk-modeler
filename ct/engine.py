"""Models: a hurricane climate fitted to forty years of best tracks, damage curves fitted to fifteen years of claims (losses
below the deductible are censored, not missing), the stochastic catalogue run through the book with policy terms and a
catastrophe treaty, exceedance-probability metrics with their sampling error, a deterministic stress test and the burn-cost
baseline that the catalogue is meant to replace.
"""
import functools
import math

import numpy as np
from scipy import optimize, stats
from scipy.special import betainc, ndtr

from . import world as W

VERSION = "ct-1.0"
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000]
CLASS_FACTOR = {"wood_frame": 1.0, "masonry": 1.12, "reinforced_concrete": 1.35, "manufactured_home": 0.78}   # engineering study: strength relative to wood frame


# ---------------------------------------------------------------- hazard

def fit_climate(obs, years=W.HISTORY_YEARS):
    """Maximum likelihood for each piece of the climate from observed landfalls."""
    n = len(obs["dp"])
    dp = np.asarray(obs["dp"])

    def nll(p):
        k, sc = np.exp(p)
        lo = math.exp(-(W.DP_MIN / sc) ** k) - math.exp(-(W.DP_MAX / sc) ** k)
        return -(stats.weibull_min.logpdf(dp, k, scale=sc).sum() - n * math.log(max(lo, 1e-300)))
    k, sc = np.exp(optimize.minimize(nll, [math.log(1.5), math.log(30)], method="Nelder-Mead").x)
    A = np.c_[np.ones(n), dp]
    coef, *_ = np.linalg.lstsq(A, np.log(obs["rmax"]), rcond=None)
    resid = np.log(obs["rmax"]) - A @ coef
    z = np.clip(np.asarray(obs["x0"]) / W.COAST_KM, 0, 1)
    tilt = optimize.minimize_scalar(lambda a: -np.log(np.maximum(1 + a * (2 * z - 1), 1e-9)).sum(), bounds=(-0.99, 0.99), method="bounded").x
    lv = np.log(obs["vt"])
    return {"rate": n / years, "dp_shape": float(k), "dp_scale": float(sc), "rmax_a": float(coef[0]), "rmax_b": float(coef[1]),
            "rmax_sd": float(max(resid.std(ddof=2), 0.05)), "vt_mu": float(lv.mean()), "vt_sd": float(max(lv.std(ddof=1), 0.05)),
            "heading_mu": float(np.mean(obs["heading"])), "heading_sd": float(max(np.std(obs["heading"], ddof=1), 5)), "x_tilt": float(tilt)}


def rate_interval(n, years, level=0.9):
    a = (1 - level) / 2
    return (stats.chi2.ppf(a, 2 * n) / 2 / years if n else 0.0, stats.chi2.ppf(1 - a, 2 * n + 2) / 2 / years)


def p_cat4(climate):
    """Chance that a landfall is category 4 or worse (sustained 58 m/s, a pressure deficit of about 85 hPa)."""
    sf = lambda d: math.exp(-(d / climate["dp_scale"]) ** climate["dp_shape"])     # noqa: E731
    return (sf(85) - sf(W.DP_MAX)) / (sf(W.DP_MIN) - sf(W.DP_MAX))


# ---------------------------------------------------------------- vulnerability

def claim_table(seed):
    """Each claim-years policy in a historical storm's footprint: the gust the modeller can reconstruct (observed track,
    geocoded location), the class, and either the reported damage ratio or 'below the deductible'."""
    h, b = W.history(seed), W.book(seed)
    o = h["observed"]
    rows = h["claims"]
    ev = np.array([r[0] for r in rows], dtype=int)
    loc = np.array([r[1] for r in rows], dtype=int)
    g, _ = W.footprint(o["x0"][ev], o["dp"][ev], o["rmax"][ev], o["vt"][ev], o["heading"][ev], b["gx"][loc], b["gy"][loc], b["elev"][loc])
    reported = np.array([r[2] is not None for r in rows])
    ratio = np.array([r[2] if r[2] is not None else np.nan for r in rows])
    ded = np.array([r[3] for r in rows])
    mod = np.where(b["year"][loc] >= 2002, 1.08, 1.0) * np.where(b["hip"][loc], 1.05, 1.0) * np.array([CLASS_FACTOR[c] for c in W.CLASSES])[b["cls"][loc]]
    return {"gust": g, "mod": mod, "reported": reported, "ratio": ratio, "ded": ded, "cls": b["cls"][loc], "storms": len(set(ev.tolist()))}


def fit_naive(ct):
    """What a fit to the claims file alone gives: least squares of reported damage ratios on the curve, as if the small
    losses that were never reported did not happen."""
    rep = ct["reported"]
    (v50, beta), _ = optimize.curve_fit(lambda g, v, bt: ndtr(np.log(g / v) / bt), ct["gust"][rep] / ct["mod"][rep], ct["ratio"][rep], p0=[76, 0.25], bounds=([20, 0.05], [300, 2]))
    return {"v50": {c: float(v50 * CLASS_FACTOR[c]) for c in W.CLASSES}, "beta": float(beta)}


def fit_vulnerability(ct):
    """Censored maximum likelihood with three parameters: wood frame's 50%-damage gust (other classes scale by the engineering
    study), the curve's spread and the concentration of the scatter around it. A policy in the footprint with no claim says
    its damage stayed under the deductible."""
    g, mod, rep, r, ded = ct["gust"], ct["mod"], ct["reported"], ct["ratio"], ct["ded"]

    def nll(p):
        v50, beta, phi = math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
        m = np.clip(ndtr(np.log(g / (v50 * mod)) / beta), 1e-6, 1 - 1e-6)
        a, bb = m * phi, (1 - m) * phi
        return -(stats.beta.logpdf(np.clip(r[rep], 1e-6, 1 - 1e-6), a[rep], bb[rep]).sum() + np.log(np.maximum(betainc(a[~rep], bb[~rep], np.minimum(ded[~rep], 1.0)), 1e-300)).sum())   # a flat deductible can exceed a small building's value: CDF 1, where betainc gives NaN
    best = min((optimize.minimize(nll, [math.log(v), math.log(0.25), math.log(6)], method="Nelder-Mead", options={"maxiter": 2000}) for v in (60, 80, 100)), key=lambda o: o.fun)
    v50, beta, phi = np.exp(best.x)
    return {"v50": {c: float(v50 * CLASS_FACTOR[c]) for c in W.CLASSES}, "beta": float(beta), "phi": float(phi)}


def curve(vuln, cls="wood_frame", gusts=None):
    gusts = np.arange(20, 91, 5) if gusts is None else gusts
    return [float(ndtr(math.log(gg / vuln["v50"][cls]) / vuln["beta"])) for gg in gusts]


@functools.lru_cache(maxsize=8)
def calibrate(seed):
    h = W.history(seed)
    climate = fit_climate(h["observed"])
    ct = claim_table(seed)
    return {"climate": climate, "vuln": fit_vulnerability(ct), "vuln_naive": fit_naive(ct),
            "storms": h["n_storms"], "claims": {"policies_in_footprints": int(len(ct["gust"])), "reported": int(ct["reported"].sum()), "storms": ct["storms"]}}


# ---------------------------------------------------------------- catalogue and metrics

def run_catalogue(b, climate, vuln, n_events, seed, coords="geocoded"):
    ev = W.sample_events(climate, n_events=n_events, rng=np.random.default_rng(seed))
    loss = W.event_losses(b, ev, vuln, coords=coords, rng=np.random.default_rng(seed + 1))
    loss["net"] = W.net_of_reinsurance(loss["gross"])
    return ev, loss


def annual(ev, per_event):
    yrs = ev["n_years"]
    occ = np.zeros(yrs)
    np.maximum.at(occ, ev["year"], per_event)
    return occ, np.bincount(ev["year"], per_event, minlength=yrs)


def pml(annual_losses, rps=RETURN_PERIODS):
    return {rp: float(np.quantile(annual_losses, 1 - 1 / rp)) for rp in rps}


def bootstrap_pml(annual_losses, rps=(100, 250), reps=100, seed=0):
    """Sampling error of the catalogue: resample whole years."""
    r = np.random.default_rng(seed)
    n = len(annual_losses)
    out = {rp: [] for rp in rps}
    for _ in range(reps):
        s = annual_losses[r.integers(0, n, n)]
        for rp in rps:
            out[rp].append(np.quantile(s, 1 - 1 / rp))
    return {rp: (float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))) for rp, v in out.items()}


def tvar(annual_losses, rp=100):
    q = np.quantile(annual_losses, 1 - 1 / rp)
    return float(annual_losses[annual_losses >= q].mean())


def return_period_of(loss, occ):
    p = (occ >= loss).mean()
    return float(1 / p) if p > 0 else None


def summarise(ev, loss):
    occ_g, agg_g = annual(ev, loss["gross"])
    occ_n, agg_n = annual(ev, loss["net"])
    yrs = ev["n_years"]
    return {"years": int(yrs), "events": int(len(ev["x0"])), "aal_gross": float(loss["gross"].sum() / yrs), "aal_net": float(loss["net"].sum() / yrs),
            "aal_ground_up": float(loss["ground_up"].sum() / yrs), "surge_share": float(loss["surge"].sum() / max(loss["gross"].sum(), 1)),
            "oep_gross": pml(occ_g), "aep_gross": pml(agg_g), "oep_net": pml(occ_n), "aep_net": pml(agg_n),
            "tvar100_gross": tvar(occ_g), "ci": bootstrap_pml(occ_g), "p_attach": float((occ_g > W.REINSURANCE["retention"]).mean()),
            "p_exhaust": float((occ_g > W.REINSURANCE["retention"] + W.REINSURANCE["limit"]).mean())}, occ_g


def burn_cost(b, seed, vuln):
    """The forty observed storms, as if they hit today's book: the average year and the worst."""
    o = W.history(seed)["observed"]
    ev = {k: np.asarray(v) for k, v in o.items()}
    loss = W.event_losses(b, ev, vuln, rng=np.random.default_rng(3))
    per_year = np.bincount(ev["year"] - ev["year"].min() if len(ev["year"]) else ev["year"], loss["gross"], minlength=W.HISTORY_YEARS)
    return {"aal_gross": float(loss["gross"].sum() / W.HISTORY_YEARS), "worst_year": float(per_year.max() if len(per_year) else 0), "storms": int(len(ev["x0"]))}


# ---------------------------------------------------------------- stress test

def stress(b, vuln, x0, dp, rmax, vt, heading, draws=200, seed=7):
    """One deterministic storm. Expected loss per location is the mean over draws of the damage scatter: with deductibles,
    loss is convex in damage, so the loss at the mean damage ratio understates it."""
    ev = {"x0": np.array([float(x0)]), "dp": np.array([float(dp)]), "rmax": np.array([float(rmax)]), "vt": np.array([float(vt)]), "heading": np.array([float(heading)])}
    d = W.event_losses(b, ev, vuln, rng=np.random.default_rng(seed), keep_location=0)["detail"]
    if d is None:
        return {"gross_mean": 0.0, "draws": np.zeros(1), "detail": None}
    r = np.random.default_rng(seed + 1)
    T = W.tiv(b)
    per_loc = np.zeros(len(d["idx"]))
    totals = []
    for _ in range(draws):
        _, gr = W.financial(b, d["idx"], W.realise(d["mean"], r), d["surge_share"], T)
        per_loc += gr
        totals.append(gr.sum())
    per_loc /= draws
    covered_surge = np.where(b["surge_covered"][d["idx"]], d["surge_share"], 0)
    return {"gross_mean": float(per_loc.sum()), "draws": np.array(totals), "detail": {**d, "gross_expected": per_loc, "surge_covered": covered_surge}}
