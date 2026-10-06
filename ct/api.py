"""Public API (modular monolith). Blueprint services map to: exposure ingest and geocoder (portfolio import, row checks,
postal-zone centroids), hazard model and vulnerability model (calibrated from history), policy engine (deductibles, limits,
surge exclusions, the catastrophe treaty), loss simulator (the stochastic catalogue as a background job), portfolio analytics
(exceedance curves, losses by town and class) and the scenario catalog (stochastic runs and deterministic stress tests).

    uvicorn ct.api:app          python -m core.jobs ct.api      # the worker
"""
import time
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"book:read", "jobs:read"}
WORK = READ | {"book:write", "models:calibrate", "scenarios:run"}
PERMISSIONS = {"viewer": READ, "analyst": WORK, "chief_risk_officer": WORK | {"book:generate", "scenarios:approve", "audit:read"}}
app = create_app("cat-risk-modeler", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
TOWN_NAMES = [t[0] for t in W.TOWNS]
POSTAL_X, POSTAL_Y = 10.0, 6.0             # postal zones: 10 km along the coast x 6 km inland, named PZ-<x cell>-<y cell>
TRUTH_EVENTS = 20000                        # the simulation-truth catalogue (true climate, true curves, true coordinates)


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def lid(t, ref):
    return uuid.uuid5(t, f"loc-{ref}")


def portfolio(c, pid=None):
    p = c.execute("SELECT * FROM portfolio WHERE portfolio_id = %s" if pid else "SELECT * FROM portfolio ORDER BY created_at LIMIT 1", [pid] if pid else []).fetchone()
    if not p:
        raise Problem(404 if pid else 409, "portfolio_not_found" if pid else "no_portfolio", "" if pid else "import a portfolio first: POST /v1/portfolios/import")
    return p


def book_from_db(c, pid):
    """The book as the models see it: geocoded coordinates, building attributes, policy terms and coverage limits."""
    rows = c.execute("""SELECT l.location_id, l.external_ref, l.town, l.x_km, l.y_km, l.geocode_precision, l.elevation_m, b.construction, b.year_built, b.roof, b.occupancy,
                               p.attributes, max(cv.limit_usd) FILTER (WHERE cv.kind = 'building') AS bldg, max(cv.limit_usd) FILTER (WHERE cv.kind = 'contents') AS contents,
                               max(cv.limit_usd) FILTER (WHERE cv.kind = 'time_element') AS bi
                          FROM location l JOIN building b USING (location_id) JOIN policy p USING (location_id) JOIN coverage cv USING (policy_id)
                         WHERE l.portfolio_id = %s AND p.status = 'in_force' GROUP BY l.location_id, b.location_id, p.policy_id ORDER BY l.x_km, l.external_ref""", [pid]).fetchall()
    if not rows:
        raise Problem(409, "empty_portfolio")
    f = lambda k, dt=float: np.array([r[k] for r in rows], dtype=dt)     # noqa: E731
    return {"ids": [r["location_id"] for r in rows], "refs": [r["external_ref"] for r in rows], "town": np.array([TOWN_NAMES.index(r["town"]) if r["town"] in TOWN_NAMES else -1 for r in rows]),
            "gx": f("x_km"), "gy": f("y_km"), "elev": f("elevation_m"), "rooftop": np.array([r["geocode_precision"] == "rooftop" for r in rows]),
            "cls": np.array([W.CLASSES.index(r["construction"]) for r in rows]), "year": f("year_built", int), "hip": np.array([r["roof"] == "hip" for r in rows]),
            "commercial": np.array([r["occupancy"] == "commercial" for r in rows]), "bldg": f("bldg"), "contents": f("contents"), "bi": f("bi"),
            "ded_pct": np.array([float(r["attributes"]["deductible_pct"]) for r in rows]), "ded_flat": np.array([float(r["attributes"]["deductible_usd"]) for r in rows]),
            "surge_covered": np.array([bool(r["attributes"]["surge_covered"]) for r in rows])}


def models(c):
    rows = {r["model_name"]: r for r in c.execute("SELECT DISTINCT ON (model_name) * FROM model_artifact ORDER BY model_name, created_at DESC")}
    if "hurricane_climate" not in rows or "wind_vulnerability" not in rows:
        raise Problem(409, "no_models", "calibrate first: POST /v1/models/calibrate")
    return rows


def money(v):
    return round(float(v))


# ---------------------------------------------------------------- exposure ingest and geocoder

class Generate(BaseModel):
    seed: int = Field(11, ge=0, le=10 ** 6)
    locations: int = Field(10000, ge=100, le=10000)


class Row(BaseModel):
    ref: str = Field(min_length=1, max_length=40)
    x_km: float | None = None
    y_km: float | None = None
    postal_zone: str | None = Field(None, max_length=20, description="when there is no rooftop geocode, e.g. PZ-39-0")
    town: str = Field("Port Ellis", max_length=40)
    elevation_m: float = Field(2.0, ge=-5, le=500)
    construction: str
    year_built: int = Field(ge=1800, le=2100)
    roof: str = Field("gable", pattern="^(hip|gable)$")
    stories: int = Field(1, ge=1, le=60)
    occupancy: str = Field("residential", pattern="^(residential|commercial)$")
    building_value: float
    contents_value: float = Field(0, ge=0)
    time_element_value: float = Field(0, ge=0)
    deductible_pct: float = Field(0.02, ge=0, le=0.5)
    deductible_usd: float = Field(0, ge=0)
    surge_covered: bool = False


class ImportIn(BaseModel):
    external_ref: str = Field(min_length=1, max_length=60)
    name: str = Field("Coastal homeowners and small commercial", max_length=120)
    generate: Generate | None = None
    rows: list[Row] | None = Field(None, max_length=5000)


@app.post("/v1/portfolios/import", status_code=202, tags=["exposure"], summary="Import a property book. `generate` loads the synthetic coastal book in a background job; `rows` adds locations to an existing book, each row checked and geocoded on its own")
def import_portfolio(body: ImportIn, ctx: Ctx = Depends(auth("book:write")), idem: str | None = IdemKey):
    if (body.generate is None) == (body.rows is None):
        raise Problem(422, "generate_or_rows", "send exactly one of generate or rows")
    if body.generate and "book:generate" not in PERMISSIONS[ctx.role]:
        raise Problem(403, "forbidden", "generating a synthetic book needs the chief risk officer")

    def work(c):
        p = c.execute("SELECT * FROM portfolio WHERE external_ref = %s", [body.external_ref]).fetchone()
        if body.generate:
            if p:
                raise Problem(409, "portfolio_exists")
            return 202, jobs.enqueue(c, ctx, "portfolio.generate", body.model_dump(include={"external_ref", "name", "generate"}))
        if not p:
            raise Problem(404, "portfolio_not_found", f"no portfolio '{body.external_ref}'")
        existing = {r["external_ref"] for r in c.execute("SELECT external_ref FROM location WHERE portfolio_id = %s", [p["portfolio_id"]])}
        accepted, rejected = [], []
        for r in body.rows:
            why, (x, y, precision) = check_row(r, existing)
            if why:
                rejected.append({"ref": r.ref, "why": why})
                continue
            existing.add(r.ref)
            accepted.append(insert_rows(c, ctx.tenant_id, p["portfolio_id"], [(r, x, y, precision)])[0])
        audit.record(c, ctx, "portfolio.rows_imported", "portfolio", p["portfolio_id"], {"accepted": len(accepted), "rejected": len(rejected)})
        return 201, {"portfolio_id": p["portfolio_id"], "accepted": len(accepted), "rejected": rejected, "geocoded": accepted}
    return run(ctx, idem, body, work)


def check_row(r, existing):
    """-> (reason or None, (x, y, precision)). The geocoder: rooftop coordinates when given, else the postal zone's centroid."""
    none = (None, None, None)
    if r.ref in existing:
        return "duplicate location reference", none
    if r.construction not in W.CLASSES:
        return f"unknown construction '{r.construction}' (expected one of {', '.join(W.CLASSES)})", none
    if r.building_value <= 0:
        return "building value must be positive", none
    if r.deductible_pct * r.building_value + r.deductible_usd >= r.building_value:
        return "deductible is not below the building value", none
    if r.year_built > 2026:
        return "built in the future", none
    if r.x_km is not None and r.y_km is not None:
        if not (0 <= r.x_km <= W.COAST_KM) or not (0 <= r.y_km <= 60):
            return "coordinates are offshore or outside the modelled coast", none
        return None, (r.x_km, r.y_km, "rooftop")
    if r.postal_zone:
        try:
            _, cx, cy = r.postal_zone.split("-")
            cx, cy = int(cx), int(cy)
        except ValueError:
            return f"postal zone '{r.postal_zone}' not recognised", none
        if not (0 <= cx < W.COAST_KM / POSTAL_X and 0 <= cy < 60 / POSTAL_Y):
            return f"postal zone '{r.postal_zone}' is not on this coast", none
        return None, ((cx + 0.5) * POSTAL_X, (cy + 0.5) * POSTAL_Y, "postal_centroid")
    return "no coordinates and no postal zone: cannot be placed", none


def insert_rows(c, t, pid, items):
    """items: (Row-like, x, y, precision). Writes location, building, policy and its three coverages."""
    loc, bld, pol, cov, out = [], [], [], [], []
    for r, x, y, precision in items:
        l, pol_id = lid(t, r.ref), uuid.uuid5(t, f"pol-{r.ref}")
        loc.append((l, t, pid, r.ref, r.town, x, y, precision, r.elevation_m))
        bld.append((l, t, r.construction, r.year_built, r.roof, r.stories, r.occupancy))
        pol.append((pol_id, t, pid, "in_force", Jsonb({"deductible_pct": r.deductible_pct, "deductible_usd": r.deductible_usd, "surge_covered": r.surge_covered, "peril": "hurricane (wind; surge only where covered)"}), l))
        cov += [(pol_id, "building", r.building_value, t), (pol_id, "contents", r.contents_value, t), (pol_id, "time_element", r.time_element_value, t)]
        out.append({"ref": r.ref, "x_km": round(x, 2), "y_km": round(y, 2), "geocode_precision": precision})
    db.load(c, "location", ["location_id", "tenant_id", "portfolio_id", "external_ref", "town", "x_km", "y_km", "geocode_precision", "elevation_m"], loc)
    db.load(c, "building", ["location_id", "tenant_id", "construction", "year_built", "roof", "stories", "occupancy"], bld)
    db.load(c, "policy", ["policy_id", "tenant_id", "portfolio_id", "status", "attributes", "location_id"], pol)
    db.load(c, "coverage", ["policy_id", "kind", "limit_usd", "tenant_id"], cov)
    return out


class _R:     # a generated row in the shape insert_rows reads
    def __init__(self, **kw):
        self.__dict__.update(kw)


@jobs.handler("portfolio.generate")
def generate_job(c, job):
    t, p = job["tenant_id"], job["payload"]
    g = p["generate"]
    b = W.book(g["seed"], g["locations"])
    pid = uuid.uuid5(t, f"portfolio-{p['external_ref']}")
    c.execute("INSERT INTO portfolio (portfolio_id, tenant_id, external_ref, name, metadata, seed) VALUES (%s,%s,%s,%s,%s,%s)",
              [pid, t, p["external_ref"], p["name"], Jsonb({"source": "synthetic coastal book", "locations": g["locations"]}), g["seed"]])
    items = []
    for i in range(len(b["x"])):
        r = _R(ref=f"L{i + 1:05d}", town=TOWN_NAMES[b["town"][i]], elevation_m=round(float(b["elev"][i]), 2), construction=W.CLASSES[b["cls"][i]], year_built=int(b["year"][i]),
               roof="hip" if b["hip"][i] else "gable", stories=int(b["stories"][i]), occupancy="commercial" if b["commercial"][i] else "residential",
               building_value=round(float(b["bldg"][i])), contents_value=round(float(b["contents"][i])), time_element_value=round(float(b["bi"][i])),
               deductible_pct=float(b["ded_pct"][i]), deductible_usd=float(b["ded_flat"][i]), surge_covered=bool(b["surge_covered"][i]))
        items.append((r, round(float(b["gx"][i]), 3), round(float(b["gy"][i]), 3), "rooftop" if b["rooftop"][i] else "postal_centroid"))
    insert_rows(c, t, pid, items)
    audit.record(c, worker_ctx(job), "portfolio.generated", "portfolio", pid, {"locations": len(items), "seed": g["seed"]})
    return {"portfolio_id": pid, "locations": len(items), "tiv": money(W.tiv(b).sum())}


@app.get("/v1/portfolios/{portfolio_id}", tags=["exposure"], summary="(+) The book: insured value by town, construction and geocode precision, and the policy terms")
def get_portfolio(portfolio_id: uuid.UUID, ctx: Ctx = Depends(auth("book:read"))):
    with db.tx(ctx.tenant_id) as c:
        p = portfolio(c, portfolio_id)
        b = book_from_db(c, portfolio_id)
    T = W.tiv(b)
    by = lambda keys, names: [{"name": n, "locations": int((keys == k).sum()), "tiv": money(T[keys == k].sum())} for k, n in enumerate(names)]     # noqa: E731
    return jsonable_encoder({"portfolio_id": portfolio_id, "name": p["name"], "external_ref": p["external_ref"], "locations": len(T), "tiv": money(T.sum()),
                             "by_town": by(b["town"], TOWN_NAMES), "by_construction": by(b["cls"], W.CLASSES),
                             "geocode": {"rooftop": int(b["rooftop"].sum()), "postal_centroid": int((~b["rooftop"]).sum())},
                             "terms": {"homeowners_with_surge_excluded": int((~b["surge_covered"] & ~b["commercial"]).sum()), "commercial": int(b["commercial"].sum()),
                                       "commercial_with_surge_cover": int((b["surge_covered"]).sum()), "deductible_2pct": int((b["ded_pct"] == 0.02).sum()), "deductible_5pct": int((b["ded_pct"] == 0.05).sum())},
                             "within_2km_of_coast_tiv": money(T[b["gy"] < 2].sum()), "reinsurance": {"type": "per-occurrence catastrophe excess of loss", "retention": W.REINSURANCE["retention"], "limit": W.REINSURANCE["limit"]}})


# ---------------------------------------------------------------- hazard and vulnerability

class CalibrateIn(BaseModel):
    portfolio_id: uuid.UUID


@app.post("/v1/models/calibrate", status_code=201, tags=["models"], summary="(+) Fit the hurricane climate to forty years of best tracks and the wind damage curve to fifteen years of claims; store both as versioned models")
def calibrate(body: CalibrateIn, ctx: Ctx = Depends(auth("models:calibrate")), idem: str | None = IdemKey):
    def work(c):
        p = portfolio(c, body.portfolio_id)
        seed = p["seed"]
        cal = E.calibrate(seed)
        h = W.history(seed)
        clim, vul = cal["climate"], cal["vuln"]
        lo, hi = E.rate_interval(cal["storms"], W.HISTORY_YEARS)
        boot = cat4_interval(h["observed"]["dp"], seed)
        ver = f"{E.VERSION}-{int(time.time() * 1000)}"
        for name, params, metrics in (("hurricane_climate", clim, {"storms": cal["storms"], "years": W.HISTORY_YEARS, "rate_90": [lo, hi], "p_cat4_90": boot}),
                                      ("wind_vulnerability", vul, {**cal["claims"], "method": "censored maximum likelihood"})):
            c.execute("INSERT INTO model_artifact (model_id, tenant_id, model_name, version, artifact_uri, feature_schema, metrics, approved, params) VALUES (%s,%s,%s,%s,%s,%s,%s,false,%s)",
                      [uuid.uuid7(), ctx.tenant_id, name, ver, f"db://model_artifact/{name}/{ver}", Jsonb({"inputs": ["best tracks"] if name == "hurricane_climate" else ["gust at location", "construction", "year built", "roof"]}), Jsonb(metrics), Jsonb(params)])
        audit.record(c, ctx, "models.calibrated", "model_artifact", ver, {"storms": cal["storms"], "claims_reported": cal["claims"]["reported"]})
        o = h["observed"]
        storms = [{"year": int(o["year"][i]), "landfall_x_km": round(float(o["x0"][i])), "category": W.category(float(W.v0(o["dp"][i]))), "pressure_deficit_hpa": round(float(o["dp"][i]), 1)} for i in range(len(o["dp"]))]
        gusts = list(range(30, 91, 5))
        return 201, {"version": ver, "history": {"years": W.HISTORY_YEARS, "storms": storms, "by_category": [sum(1 for s in storms if s["category"] == k) for k in range(6)]},
                     "climate": {"rate_per_year": round(clim["rate"], 3), "rate_90": [round(lo, 3), round(hi, 3)], "p_cat4_or_worse": round(E.p_cat4(clim), 4), "p_cat4_90": [round(x, 4) for x in boot],
                                 "dp_weibull": [round(clim["dp_shape"], 2), round(clim["dp_scale"], 1)], "east_tilt": round(clim["x_tilt"], 2)},
                     "vulnerability": {**{k: round(v, 3) for k, v in cal["claims"].items()}, "v50_wood_frame": round(vul["v50"]["wood_frame"], 1), "beta": round(vul["beta"], 3), "phi": round(vul["phi"], 2),
                                       "naive_v50_wood_frame": round(cal["vuln_naive"]["v50"]["wood_frame"], 1), "naive_beta": round(cal["vuln_naive"]["beta"], 3),
                                       "curve": [{"gust": g, "fitted": round(f, 4), "naive": round(n, 4), "truth": round(t, 4)} for g, f, n, t in zip(gusts, E.curve(vul, gusts=gusts), E.curve(cal["vuln_naive"], gusts=gusts), E.curve(W.TRUE_VULN, gusts=gusts))]},
                     "simulation_truth": {"what": "the climate and curve the history came from, which the fits never see", "rate_per_year": W.TRUE_CLIMATE["rate"], "p_cat4_or_worse": round(E.p_cat4(W.TRUE_CLIMATE), 4), "v50_wood_frame": W.TRUE_VULN["v50"]["wood_frame"], "beta": W.TRUE_VULN["beta"]},
                     "how": "climate by maximum likelihood (Poisson rate, truncated Weibull pressure deficit, log-linear radius, lognormal speed, normal heading, linear landfall density); the damage curve by censored maximum likelihood: a reported claim gives its damage ratio, a policy in the footprint with no claim says the damage stayed under its deductible; the naive fit is least squares on reported claims alone"}
    return run(ctx, idem, body, work)


def cat4_interval(dp, seed, reps=200):
    r = np.random.default_rng(seed)
    dp = np.asarray(dp)
    vals = []
    for _ in range(reps):
        s = dp[r.integers(0, len(dp), len(dp))]
        vals.append(E.p_cat4(E.fit_climate({"dp": s, "rmax": np.full(len(s), 30.0) * np.exp(r.normal(0, .1, len(s))), "x0": np.full(len(s), 300.0), "vt": np.full(len(s), 5.0) * np.exp(r.normal(0, .1, len(s))), "heading": r.normal(0, 10, len(s))})))
    return [float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))]


# ---------------------------------------------------------------- loss simulator

class RunIn(BaseModel):
    portfolio_id: uuid.UUID
    events: int = Field(50000, ge=1000, le=200000)
    seed: int = Field(2026, ge=0, le=10 ** 6)


@app.post("/v1/scenarios/run", status_code=202, tags=["scenarios"], summary="Run the stochastic catalogue through the book in a background job: events from the calibrated climate, damage from the calibrated curves with secondary uncertainty, policy terms, the treaty")
def run_scenario(body: RunIn, ctx: Ctx = Depends(auth("scenarios:run")), idem: str | None = IdemKey):
    def work(c):
        portfolio(c, body.portfolio_id)
        m = models(c)
        rid = uuid.uuid7()
        c.execute("INSERT INTO scenario_run (run_id, tenant_id, portfolio_id, kind, status, params, created_by) VALUES (%s,%s,%s,'stochastic','queued',%s,%s)",
                  [rid, ctx.tenant_id, body.portfolio_id, Jsonb({"events": body.events, "seed": body.seed, "models": {k: v["version"] for k, v in m.items()}}), ctx.actor_id])
        return 202, {**jobs.enqueue(c, ctx, "scenario.run", {"run_id": str(rid), **body.model_dump(mode="json")}), "run_id": rid}
    return run(ctx, idem, body, work)


@jobs.handler("scenario.run")
def run_job(c, job):
    p = job["payload"]
    t, rid = job["tenant_id"], uuid.UUID(p["run_id"])
    port = portfolio(c, uuid.UUID(p["portfolio_id"]))
    m = models(c)
    b = book_from_db(c, port["portfolio_id"])
    t0 = time.perf_counter()
    ev, loss = E.run_catalogue(b, m["hurricane_climate"]["params"], m["wind_vulnerability"]["params"], p["events"], p["seed"])
    secs = time.perf_counter() - t0
    s, occ = E.summarise(ev, loss)
    cats = np.array([W.category(v) for v in W.v0(ev["dp"])])
    db.load(c, "hazard_event", ["run_id", "event_no", "year", "landfall_x_km", "pressure_deficit_hpa", "rmax_km", "forward_speed_ms", "heading_deg", "category", "tenant_id"],
            [(rid, i, int(ev["year"][i]), float(ev["x0"][i]), float(ev["dp"][i]), float(ev["rmax"][i]), float(ev["vt"][i]), float(ev["heading"][i]), int(cats[i]), t) for i in range(len(cats))])
    db.load(c, "loss_sample", ["run_id", "event_no", "year", "ground_up", "gross", "net", "surge", "tenant_id"],
            [(rid, i, int(ev["year"][i]), float(loss["ground_up"][i]), float(loss["gross"][i]), float(loss["net"][i]), float(loss["surge"][i]), t) for i in range(len(cats))])
    ep = [(rid, basis, persp, rp, s[f"{basis}_{persp}"][rp], t) for basis in ("oep", "aep") for persp in ("gross", "net") for rp in E.RETURN_PERIODS]
    db.load(c, "ep_curve", ["run_id", "basis", "perspective", "return_period", "loss", "tenant_id"], ep)
    seed = port["seed"]
    truth = truth_summary(seed)
    burn = E.burn_cost(W.book(seed), seed, m["wind_vulnerability"]["params"])
    T = W.tiv(b)
    contrib = lambda keys, names: [{"name": n, "aal": money(loss["by_location"][keys == k].sum() / s["years"]), "tiv": money(T[keys == k].sum())} for k, n in enumerate(names)]     # noqa: E731
    metrics = {**{k: v for k, v in s.items() if k not in ("ci",)}, "ci_90": {str(k): v for k, v in s["ci"].items()}, "seconds": round(secs, 1), "tiv": money(T.sum()),
               "aal_by_category": [money(loss["gross"][cats == k].sum() / s["years"]) for k in range(6)], "events_by_category": np.bincount(cats, minlength=6).tolist(),
               "aal_by_town": contrib(b["town"], TOWN_NAMES), "aal_by_construction": contrib(b["cls"], W.CLASSES),
               "aal_by_geocode": {"rooftop": money(loss["by_location"][b["rooftop"]].sum() / s["years"]), "postal_centroid": money(loss["by_location"][~b["rooftop"]].sum() / s["years"])},
               "burn_cost": burn, "simulation_truth": truth}
    metrics = jsonable_encoder(metrics)
    c.execute("UPDATE scenario_run SET status = 'completed', metrics = %s WHERE run_id = %s", [Jsonb(metrics), rid])
    audit.record(c, worker_ctx(job), "scenario.completed", "scenario_run", rid, {"events": s["events"], "years": s["years"], "aal_gross": money(s["aal_gross"]), "pml_250": money(s["oep_gross"][250])})
    return {"run_id": rid, "events": s["events"], "years": s["years"], "seconds": round(secs, 1), "aal_gross": money(s["aal_gross"]), "pml_100": money(s["oep_gross"][100]), "pml_250": money(s["oep_gross"][250])}


_truth = {}


def truth_summary(seed):
    """The simulation truth: a catalogue from the true climate through the true curves at the true coordinates."""
    if seed not in _truth:
        ev, loss = E.run_catalogue(W.book(seed), W.TRUE_CLIMATE, W.TRUE_VULN, TRUTH_EVENTS, seed + 99, coords="true")
        s, _ = E.summarise(ev, loss)
        _truth[seed] = {"what": f"{TRUTH_EVENTS:,} events from the true climate, true damage curves and true coordinates; no model sees it", "events": s["events"], "years": s["years"],
                        "aal_gross": s["aal_gross"], "oep_gross": s["oep_gross"], "oep_net": s["oep_net"], "aal_net": s["aal_net"]}
    return _truth[seed]


def scenario(c, run_id):
    r = c.execute("SELECT * FROM scenario_run WHERE run_id = %s", [run_id]).fetchone()
    if not r:
        raise Problem(404, "run_not_found")
    return r


@app.get("/v1/scenarios/{run_id}/losses", tags=["scenarios"], summary="Where the losses come from: the largest events, average annual loss by category, town, construction and geocode precision")
def losses(run_id: uuid.UUID, top: int = 10, ctx: Ctx = Depends(auth("book:read"))):
    with db.tx(ctx.tenant_id) as c:
        r = scenario(c, run_id)
        if r["kind"] != "stochastic" or r["status"] != "completed":
            raise Problem(409, "run_not_ready", f"{r['kind']} run is {r['status']}")
        ev = c.execute("""SELECT l.event_no, l.year, l.gross, l.net, l.surge, h.landfall_x_km, h.pressure_deficit_hpa, h.rmax_km, h.category
                            FROM loss_sample l JOIN hazard_event h USING (run_id, event_no) WHERE l.run_id = %s ORDER BY l.gross DESC LIMIT %s""", [run_id, min(top, 100)]).fetchall()
    m = r["metrics"]
    return jsonable_encoder({"run_id": run_id, "status": r["status"], "approved": r["approved_by"] is not None, "events": m["events"], "years": m["years"], "aal_gross": m["aal_gross"],
                             "top_events": [{"event_no": e["event_no"], "year": e["year"], "gross": money(e["gross"]), "net": money(e["net"]), "surge_share": round(e["surge"] / e["gross"], 3) if e["gross"] else 0,
                                             "landfall_x_km": round(e["landfall_x_km"]), "nearest_town": min(W.TOWNS, key=lambda tw: abs(tw[1] - e["landfall_x_km"]))[0], "category": e["category"], "rmax_km": round(e["rmax_km"])} for e in ev],
                             "aal_by_category": m["aal_by_category"], "events_by_category": m["events_by_category"], "aal_by_town": m["aal_by_town"], "aal_by_construction": m["aal_by_construction"], "aal_by_geocode": m["aal_by_geocode"], "surge_share": m["surge_share"]})


@app.get("/v1/portfolios/{portfolio_id}/ep-curve", tags=["analytics"], summary="Exceedance-probability curves (occurrence and aggregate, gross and net of the treaty), PML with its sampling interval, TVaR, AAL; from the approved run unless run_id is given")
def ep_curve(portfolio_id: uuid.UUID, run_id: uuid.UUID | None = None, ctx: Ctx = Depends(auth("book:read"))):
    with db.tx(ctx.tenant_id) as c:
        portfolio(c, portfolio_id)
        r = scenario(c, run_id) if run_id else c.execute("""SELECT * FROM scenario_run WHERE portfolio_id = %s AND kind = 'stochastic' AND status = 'completed'
                                                             ORDER BY approved_at DESC NULLS LAST, created_at DESC LIMIT 1""", [portfolio_id]).fetchone()
        if not r or r["kind"] != "stochastic" or r["status"] != "completed":
            raise Problem(409, "no_completed_run", "run the catalogue first: POST /v1/scenarios/run")
        pts = c.execute("SELECT basis, perspective, return_period, loss FROM ep_curve WHERE run_id = %s ORDER BY basis, perspective, return_period", [r["run_id"]]).fetchall()
    m = r["metrics"]
    curves = {}
    for p in pts:
        curves.setdefault(f"{p['basis']}_{p['perspective']}", []).append({"return_period": p["return_period"], "loss": money(p["loss"])})
    tr = m["simulation_truth"]
    return jsonable_encoder({"portfolio_id": portfolio_id, "run_id": r["run_id"], "approved": r["approved_by"] is not None, "events": m["events"], "years": m["years"], "tiv": m["tiv"],
                             "aal_gross": money(m["aal_gross"]), "aal_net": money(m["aal_net"]), "aal_ground_up": money(m["aal_ground_up"]), "tvar100_gross": money(m["tvar100_gross"]),
                             "pml": {rp: {"gross": money(m["oep_gross"][str(rp)]), "net": money(m["oep_net"][str(rp)]), "aggregate_gross": money(m["aep_gross"][str(rp)]),
                                          **({"ci_90": [money(x) for x in m["ci_90"][str(rp)]]} if str(rp) in m["ci_90"] else {})} for rp in E.RETURN_PERIODS},
                             "curves": curves, "treaty": {**W.REINSURANCE, "p_attach": m["p_attach"], "p_exhaust": m["p_exhaust"]},
                             "burn_cost": {k: money(v) if k != "storms" else v for k, v in m["burn_cost"].items()},
                             "simulation_truth": {"what": tr["what"], "aal_gross": money(tr["aal_gross"]), "aal_net": money(tr["aal_net"]), "pml": {rp: money(tr["oep_gross"][str(rp)]) for rp in E.RETURN_PERIODS}}})


class ApproveIn(BaseModel):
    note: str = Field(min_length=3, max_length=500)


@app.post("/v1/scenarios/{run_id}/approve", tags=["scenarios"], summary="(+) Sign off a stochastic run as the published view of risk; a second person must approve, never the one who ran it")
def approve(run_id: uuid.UUID, body: ApproveIn, ctx: Ctx = Depends(auth("scenarios:run")), idem: str | None = IdemKey):
    def work(c):
        r = scenario(c, run_id)
        if r["kind"] != "stochastic" or r["status"] != "completed":
            raise Problem(409, "run_not_ready")
        if r["created_by"] == ctx.actor_id:
            raise Problem(409, "four_eyes", "the person who ran the catalogue cannot approve it")
        if "scenarios:approve" not in PERMISSIONS[ctx.role]:
            raise Problem(403, "forbidden", "approval needs the chief risk officer")
        if r["approved_by"]:
            raise Problem(409, "already_approved")
        c.execute("UPDATE scenario_run SET approved_by = %s, approved_at = now() WHERE run_id = %s", [ctx.actor_id, run_id])
        c.execute("UPDATE model_artifact SET approved = true WHERE version IN (%s, %s)", [r["params"]["models"]["hurricane_climate"], r["params"]["models"]["wind_vulnerability"]])
        audit.record(c, ctx, "scenario.approved", "scenario_run", run_id, {"note": body.note, "pml_250": money(r["metrics"]["oep_gross"]["250"])})
        return 200, {"run_id": run_id, "approved_by": ctx.actor_name, "pml_250": money(r["metrics"]["oep_gross"]["250"]), "models_approved": list(r["params"]["models"].values())}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- stress tests

class StressIn(BaseModel):
    portfolio_id: uuid.UUID
    name: str = Field("Category 4 landfall", max_length=80)
    town: str | None = Field("Port Ellis", description="aim the landfall at a town's centre, or give landfall_x_km")
    landfall_x_km: float | None = Field(None, ge=0, le=W.COAST_KM)
    pressure_deficit_hpa: float = Field(90, ge=W.DP_MIN, le=W.DP_MAX)
    rmax_km: float = Field(22, ge=8, le=90)
    forward_speed_ms: float = Field(5, ge=1.5, le=15)
    heading_deg: float = Field(0, ge=-70, le=70)
    shifts_km: list[float] = Field([-40, -20, -10, 0, 10, 20, 40], max_length=15)


@app.post("/v1/stress-tests", status_code=201, tags=["scenarios"], summary="A deterministic storm through the book: mean loss and its spread from damage uncertainty, net of the treaty, where it falls on the occurrence curve, the losses on a map, and how much it matters exactly where it lands")
def stress_test(body: StressIn, ctx: Ctx = Depends(auth("scenarios:run")), idem: str | None = IdemKey):
    def work(c):
        port = portfolio(c, body.portfolio_id)
        m = models(c)
        if body.landfall_x_km is None:
            if body.town not in TOWN_NAMES:
                raise Problem(422, "unknown_town", f"one of {', '.join(TOWN_NAMES)}")
            x0 = W.TOWNS[TOWN_NAMES.index(body.town)][1]
        else:
            x0 = body.landfall_x_km
        b = book_from_db(c, port["portfolio_id"])
        vul = m["wind_vulnerability"]["params"]
        args = (body.pressure_deficit_hpa, body.rmax_km, body.forward_speed_ms, body.heading_deg)
        st = E.stress(b, vul, x0, *args)
        d = st["detail"]
        draws = st["draws"]
        gross = st["gross_mean"]
        net = float(W.net_of_reinsurance(np.array([gross]))[0])
        last = c.execute("SELECT run_id, metrics FROM scenario_run WHERE portfolio_id = %s AND kind = 'stochastic' AND status = 'completed' ORDER BY created_at DESC LIMIT 1", [port["portfolio_id"]]).fetchone()
        rp = None
        if last:
            n_exceed = c.execute("SELECT count(*) AS n FROM (SELECT year FROM loss_sample WHERE run_id = %s GROUP BY year HAVING max(gross) >= %s) y", [last["run_id"], gross]).fetchone()["n"]
            rp = {"years": last["metrics"]["years"], "years_at_or_above": n_exceed, "return_period": round(last["metrics"]["years"] / n_exceed) if n_exceed else None}
        rid = uuid.uuid7()
        sev = int(W.category(float(W.v0(body.pressure_deficit_hpa))))
        T = W.tiv(b)
        by_town, by_cls, cells = [], [], {}
        if d is not None:
            idx, gl = d["idx"], d["gross_expected"]
            by_town = [{"name": n, "gross": money(gl[b["town"][idx] == k].sum()), "tiv_in_footprint": money(T[idx][b["town"][idx] == k].sum())} for k, n in enumerate(TOWN_NAMES) if (b["town"][idx] == k).any()]
            by_cls = [{"name": n, "gross": money(gl[b["cls"][idx] == k].sum()), "mean_damage": round(float(np.average(d["mean"][b["cls"][idx] == k], weights=b["bldg"][idx][b["cls"][idx] == k])), 4) if (b["cls"][idx] == k).any() else 0} for k, n in enumerate(W.CLASSES)]
            for j, i in enumerate(idx):
                key = (int(b["gx"][i] // 4), int(b["gy"][i] // 2))
                cell = cells.setdefault(key, [0.0, 0.0, 0])
                cell[0] += gl[j]
                cell[1] += T[i]
                cell[2] += 1
            db.load(c, "scenario_run", ["run_id", "tenant_id", "portfolio_id", "kind", "status", "params", "metrics", "created_by"],
                    [(rid, ctx.tenant_id, port["portfolio_id"], "stress", "completed", Jsonb({**body.model_dump(mode="json"), "landfall_x_km": x0, "model": m["wind_vulnerability"]["version"]}), Jsonb({"gross_mean": gross, "net": net}), ctx.actor_id)])
            db.load(c, "damage_sample", ["run_id", "location_id", "gust_ms", "surge_depth_m", "mean_damage_ratio", "gross_loss", "tenant_id"],
                    [(rid, b["ids"][i], float(d["gust"][j]), float(d["depth"][j]), float(d["mean"][j]), float(gl[j]), ctx.tenant_id) for j, i in enumerate(idx)])
        shifts = [{"shift_km": s, "landfall_x_km": round(x0 + s), "gross": money(E.stress(b, vul, x0 + s, *args, draws=40)["gross_mean"])} for s in body.shifts_km]
        tb = W.book(port["seed"])
        truth = E.stress({**tb, "gx": tb["x"], "gy": tb["y"]}, W.TRUE_VULN, x0, *args, draws=100)
        surge_total = float((d["gross_expected"] * d["surge_covered"]).sum()) if d is not None else 0.0
        uninsured_surge = 0.0
        if d is not None:
            excluded = ~b["surge_covered"][d["idx"]]
            gu_all, _ = W.financial({**b, "surge_covered": np.ones(len(b["gx"]), bool)}, d["idx"], d["mean"], d["surge_share"], T)
            gu_wind, _ = W.financial(b, d["idx"], d["mean"], d["surge_share"], T)
            uninsured_surge = float((gu_all - gu_wind)[excluded].sum())
        audit.record(c, ctx, "stress.run", "scenario_run", rid, {"name": body.name, "landfall_x_km": x0, "category": sev, "gross": money(gross)})
        return 201, {"run_id": rid, "name": body.name, "landfall_x_km": x0, "town": body.town if body.landfall_x_km is None else None, "category": sev, "sustained_wind_ms": round(float(W.v0(body.pressure_deficit_hpa)), 1),
                     "locations_hit": int(len(d["idx"])) if d is not None else 0, "tiv_in_footprint": money(T[d["idx"]].sum()) if d is not None else 0,
                     "gross_mean": money(gross), "gross_p10": money(np.quantile(draws, 0.1)), "gross_p50": money(np.quantile(draws, 0.5)), "gross_p90": money(np.quantile(draws, 0.9)),
                     "net_of_treaty": money(net), "treaty_recovery": money(gross - net), "surge_share": round(surge_total / gross, 3) if gross else 0, "uninsured_surge_ground_up": money(uninsured_surge),
                     "on_the_occurrence_curve": rp, "by_town": by_town, "by_construction": by_cls,
                     "map": [{"x": (k[0] + 0.5) * 4, "y": (k[1] + 0.5) * 2, "gross": money(v[0]), "damage": round(v[0] / v[1], 4) if v[1] else 0, "locations": v[2]} for k, v in sorted(cells.items())],
                     "shifts": shifts, "simulation_truth": {"what": "the same storm through the true damage curves at the true coordinates", "gross_mean": money(truth["gross_mean"])},
                     "how": "the storm's peak gust and surge at every geocoded location, the calibrated damage curve, policy terms (deductible, limit, surge only where covered); expected loss and its interval from 200 draws of the damage scatter (with deductibles, the loss at the average damage would understate it); the treaty per occurrence"}
    return run(ctx, idem, body, work)
