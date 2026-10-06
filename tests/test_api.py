"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, jobs, scenario

from .conftest import bearer

CRO, ANALYST, VIEWER, OTHER = bearer("coastal", "chief_risk_officer"), bearer("coastal", "analyst"), bearer("coastal", "viewer"), bearer("other", "chief_risk_officer")


def test_nothing_runs_before_a_book_and_models(client):
    assert client.post("/v1/models/calibrate", headers=ANALYST, json={"portfolio_id": "00000000-0000-0000-0000-000000000000"}).json()["code"] == "portfolio_not_found"
    assert client.post("/v1/portfolios/import", headers=ANALYST, json={"external_ref": "X", "generate": {"seed": 1}}).status_code == 403
    assert client.post("/v1/portfolios/import", headers=CRO, json={"external_ref": "X"}).json()["code"] == "generate_or_rows"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    assert demo["gen"]["result"]["locations"] == 10000 and one("SELECT count(*) AS n FROM coverage")["n"] == 3 * 10002
    assert demo["rows"]["accepted"] == 2 and len(demo["rows"]["rejected"]) == 3 and demo["rows"]["geocoded"][1]["geocode_precision"] == "postal_centroid"
    assert demo["book"]["locations"] == 10002 and demo["book"]["geocode"]["postal_centroid"] > 1000
    cal = demo["cal"]
    assert cal["history"]["years"] == 40 and len(cal["history"]["storms"]) == 14 and one("SELECT count(*) AS n FROM model_artifact")["n"] == 2
    assert cal["climate"]["rate_90"][0] < cal["simulation_truth"]["rate_per_year"] < cal["climate"]["rate_90"][1]
    low = cal["vulnerability"]["curve"][3]     # 45 m/s
    assert low["naive"] > 3 * low["truth"] and abs(low["fitted"] - low["truth"]) < 0.01
    run = demo["cat"]["result"]
    assert run["events"] == 50000 and one("SELECT count(*) AS n FROM loss_sample")["n"] == 50000 and one("SELECT count(*) AS n FROM hazard_event")["n"] == 50000
    ep = demo["ep"]
    assert ep["pml"]["100"]["ci_90"][0] <= ep["pml"]["100"]["gross"] <= ep["pml"]["100"]["ci_90"][1]
    assert ep["pml"]["250"]["net"] <= ep["pml"]["250"]["gross"] and ep["aal_net"] < ep["aal_gross"] < ep["aal_ground_up"]
    assert 0.7 < ep["pml"]["250"]["gross"] / ep["simulation_truth"]["pml"]["250"] < 1.3
    assert demo["losses"]["top_events"][0]["category"] >= 3 and sum(t["aal"] for t in demo["losses"]["aal_by_town"]) == pytest.approx(ep["aal_gross"], rel=0.01)
    st = demo["stress"]
    assert st["category"] == 4 and st["gross_p10"] < st["gross_mean"] < st["gross_p90"] and st["treaty_recovery"] == 160_000_000
    assert st["on_the_occurrence_curve"]["return_period"] > 100 and st["uninsured_surge_ground_up"] > 0
    assert max(st["shifts"], key=lambda s: s["gross"])["shift_km"] < 0          # landing west puts the town on the strong right side
    assert one("SELECT count(*) AS n FROM damage_sample")["n"] == st["locations_hit"]
    assert demo["self"]["code"] == "four_eyes" and demo["approve"]["pml_250"] == run["pml_250"] and one("SELECT count(*) AS n FROM model_artifact WHERE approved")["n"] == 2
    assert demo["audit"]["chain_valid"] and {"portfolio.generated", "portfolio.rows_imported", "models.calibrated", "scenario.completed", "stress.run", "scenario.approved"} <= {e["action"] for e in demo["audit"]["events"]}


def test_validation_replay_and_approval_rules(client, demo):
    pid = demo["gen"]["result"]["portfolio_id"]
    rid = demo["cat"]["result"]["run_id"]
    assert client.post("/v1/stress-tests", headers=ANALYST, json={"portfolio_id": pid, "town": "Atlantis"}).json()["code"] == "unknown_town"
    assert client.post("/v1/stress-tests", headers=ANALYST, json={"portfolio_id": pid, "pressure_deficit_hpa": 400}).status_code == 422
    assert client.post(f"/v1/scenarios/{rid}/approve", headers=CRO, json={"note": "again"}).json()["code"] == "already_approved"
    assert client.post(f"/v1/scenarios/{rid}/approve", headers=VIEWER, json={"note": "me"}).status_code == 403
    h = {**ANALYST, "Idempotency-Key": "stress-replay"}
    body = {"portfolio_id": pid, "town": "Bayport", "pressure_deficit_hpa": 50, "shifts_km": []}
    a, b = client.post("/v1/stress-tests", headers=h, json=body), client.post("/v1/stress-tests", headers=h, json=body)
    assert a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true"
    assert client.post("/v1/stress-tests", headers=h, json={**body, "pressure_deficit_hpa": 60}).json()["code"] == "idempotency_key_reused"
    assert client.get(f"/v1/portfolios/{pid}/ep-curve", headers=VIEWER).json()["approved"] is True


def test_tenant_isolation_and_append_only_audit(client, demo):
    pid, rid = demo["gen"]["result"]["portfolio_id"], demo["cat"]["result"]["run_id"]
    assert client.get(f"/v1/portfolios/{pid}", headers=OTHER).status_code == 404
    assert client.get(f"/v1/scenarios/{rid}/losses", headers=OTHER).status_code == 404
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Harbor Re%%'").fetchone()["tenant_id"]
    assert one("SELECT count(*) AS n FROM loss_sample", tenant=other)["n"] == 0 and one("SELECT count(*) AS n FROM location", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
