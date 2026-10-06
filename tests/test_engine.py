"""The physics, the policy terms, the fits and the exceedance metrics, without a database."""
import numpy as np

from ct import engine as E, world as W


def test_wind_field_shape():
    X = np.array([400.0, 400.0 + 10, 400.0 + 60, 400.0 - 60, 400.0, 400.0])
    Y = np.array([0.5, 0.5, 0.5, 0.5, 20.0, 50.0])
    g, depth = W.footprint(400.0, 90.0, 20.0, 5.0, 0.0, X, Y, np.full(6, 1.0))
    eyewall = W.GUST * (W.v0(90.0)) * (0.85 + 0.15 * np.exp(-0.25))
    assert abs(g[0] - eyewall) < 1.0 and g[1] > g[0]          # inside the radius the eyewall passes over; right of track is stronger
    assert g[2] > g[3] and g[2] < g[1]                         # right beats left at the same distance; wind falls off outward
    assert g[0] > g[4] > g[5]                                  # decays inland
    assert depth[1] > 0 and depth[3] == 0 and depth[5] == 0    # surge on the right at the coast only


def test_catalogue_counts_and_landfall_density():
    ev = W.sample_events(W.TRUE_CLIMATE, n_events=20000, rng=np.random.default_rng(0))
    assert len(ev["x0"]) == 20000 and abs(20000 / ev["n_years"] - W.TRUE_CLIMATE["rate"]) < 0.02
    z = ev["x0"] / W.COAST_KM
    assert abs(z.mean() - (0.5 + W.TRUE_CLIMATE["x_tilt"] / 6)) < 0.01      # mean of f(z) = 1 + a(2z - 1) is 1/2 + a/6
    assert ev["dp"].min() >= W.DP_MIN and ev["dp"].max() <= W.DP_MAX


def test_policy_terms():
    b = W.book(11, 400)
    idx = np.arange(len(b["x"]))
    gu, gross = W.financial(b, idx, np.zeros(len(idx)), np.zeros(len(idx)))
    assert gross.sum() == 0
    gu, gross = W.financial(b, idx, np.ones(len(idx)), np.zeros(len(idx)))
    assert (gross <= W.tiv(b)[idx] + 1e-6).all() and (gross <= gu).all()
    _, wind_only = W.financial(b, idx, np.full(len(idx), 0.3), np.ones(len(idx)))     # all of the damage is surge
    assert (wind_only[~b["surge_covered"]] == 0).all() and (wind_only[b["surge_covered"]] > 0).any()


def test_fits_recover_the_truth_when_the_wind_is_known():
    long = W.sample_events(W.TRUE_CLIMATE, n_years=3000, rng=np.random.default_rng(3))
    clim = E.fit_climate(long, years=3000)
    assert abs(clim["rate"] - W.TRUE_CLIMATE["rate"]) < 0.03 and abs(E.p_cat4(clim) - E.p_cat4(W.TRUE_CLIMATE)) < 0.006
    ct = E.claim_table(21)
    h, b = W.history(21), W.book(21)
    t = h["truth"]
    ev, loc = np.array([c[0] for c in h["claims"]]), np.array([c[1] for c in h["claims"]])
    true_gust, _ = W.footprint(t["x0"][ev], t["dp"][ev], t["rmax"][ev], t["vt"][ev], t["heading"][ev], b["x"][loc], b["y"][loc], b["elev"][loc])
    fit = E.fit_vulnerability({**ct, "gust": true_gust})
    assert abs(fit["v50"]["wood_frame"] / W.TRUE_VULN["v50"]["wood_frame"] - 1) < 0.05 and abs(fit["beta"] - W.TRUE_VULN["beta"]) < 0.04
    naive = E.fit_naive(ct)
    assert E.curve(naive, gusts=[45])[0] > 3 * E.curve(W.TRUE_VULN, gusts=[45])[0]      # ignoring unreported small losses overstates light damage


def test_exceedance_metrics_and_treaty():
    b = W.book(11, 2000)
    ev, loss = E.run_catalogue(b, W.TRUE_CLIMATE, W.TRUE_VULN, 5000, 4)
    s, occ = E.summarise(ev, loss)
    for rp in E.RETURN_PERIODS:
        assert s["aep_gross"][rp] >= s["oep_gross"][rp] - 1e-6 and s["oep_net"][rp] <= s["oep_gross"][rp] + 1e-6
    layer = loss["gross"] - loss["net"]
    assert (layer <= W.REINSURANCE["limit"] + 1e-6).all() and (layer[loss["gross"] <= W.REINSURANCE["retention"]] == 0).all()
    lo, hi = s["ci"][100]
    assert lo <= s["oep_gross"][100] <= hi and s["tvar100_gross"] >= s["oep_gross"][100]
    assert abs(E.return_period_of(s["oep_gross"][100], occ) - 100) < 10


def test_stress_expected_loss_is_above_the_loss_at_mean_damage():
    b = W.book(11)
    st = E.stress(b, W.TRUE_VULN, 395, 90, 22, 5, 0, draws=60)
    d = st["detail"]
    _, at_mean = W.financial(b, d["idx"], d["mean"], d["surge_share"])
    assert st["gross_mean"] > at_mean.sum() and np.quantile(st["draws"], 0.1) < st["gross_mean"] < np.quantile(st["draws"], 0.9)
