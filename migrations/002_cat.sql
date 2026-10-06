-- Domain model. Names follow the blueprint's schema; additions are marked (+). Points are x/y kilometres on the synthetic coast
-- (x along the coast, y inland), not PostGIS.

CREATE TABLE portfolio (portfolio_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, external_ref text, name text, metadata jsonb NOT NULL DEFAULT '{}', created_at timestamptz NOT NULL DEFAULT now(), seed int NOT NULL, UNIQUE (tenant_id, external_ref));   -- (+) seed of the synthetic book
SELECT enable_tenant_rls('portfolio');

CREATE TABLE location (location_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, portfolio_id uuid NOT NULL REFERENCES portfolio, external_ref text NOT NULL, town text NOT NULL, x_km double precision NOT NULL, y_km double precision NOT NULL, geocode_precision text NOT NULL CHECK (geocode_precision IN ('rooftop', 'postal_centroid')), elevation_m double precision NOT NULL, UNIQUE (portfolio_id, external_ref));
SELECT enable_tenant_rls('location');

CREATE TABLE building (location_id uuid PRIMARY KEY REFERENCES location, tenant_id uuid NOT NULL REFERENCES tenants, construction text NOT NULL, year_built int NOT NULL, roof text NOT NULL, stories int NOT NULL, occupancy text NOT NULL);
SELECT enable_tenant_rls('building');

CREATE TABLE policy (policy_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, portfolio_id uuid REFERENCES portfolio, status text NOT NULL, attributes jsonb NOT NULL DEFAULT '{}', observed_at timestamptz, created_at timestamptz NOT NULL DEFAULT now(), location_id uuid NOT NULL REFERENCES location);   -- (+) location
SELECT enable_tenant_rls('policy');

CREATE TABLE coverage (policy_id uuid NOT NULL REFERENCES policy, kind text NOT NULL CHECK (kind IN ('building', 'contents', 'time_element')), limit_usd numeric NOT NULL CHECK (limit_usd >= 0), tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (policy_id, kind));
SELECT enable_tenant_rls('coverage');

-- The blueprint's model_artifact: calibrated hazard and vulnerability models, versioned, with their fit diagnostics.
CREATE TABLE model_artifact (model_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, model_name text NOT NULL, version text NOT NULL, artifact_uri text NOT NULL, feature_schema jsonb NOT NULL, metrics jsonb NOT NULL DEFAULT '{}', approved boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now(), params jsonb NOT NULL, UNIQUE (tenant_id, model_name, version));   -- (+) params
SELECT enable_tenant_rls('model_artifact');

CREATE TABLE scenario_run (run_id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, portfolio_id uuid NOT NULL REFERENCES portfolio, kind text NOT NULL CHECK (kind IN ('stochastic', 'stress')), status text NOT NULL, params jsonb NOT NULL, metrics jsonb NOT NULL DEFAULT '{}', created_by uuid NOT NULL, approved_by uuid, approved_at timestamptz, created_at timestamptz NOT NULL DEFAULT now());   -- (+) approval: the run a second person signs off is the published view of risk
SELECT enable_tenant_rls('scenario_run');

CREATE TABLE hazard_event (run_id uuid NOT NULL REFERENCES scenario_run ON DELETE CASCADE, event_no int NOT NULL, year int NOT NULL, landfall_x_km real NOT NULL, pressure_deficit_hpa real NOT NULL, rmax_km real NOT NULL, forward_speed_ms real NOT NULL, heading_deg real NOT NULL, category smallint NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (run_id, event_no));
SELECT enable_tenant_rls('hazard_event');

CREATE TABLE loss_sample (run_id uuid NOT NULL REFERENCES scenario_run ON DELETE CASCADE, event_no int NOT NULL, year int NOT NULL, ground_up double precision NOT NULL, gross double precision NOT NULL, net double precision NOT NULL, surge double precision NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (run_id, event_no));
SELECT enable_tenant_rls('loss_sample');
CREATE INDEX loss_sample_gross ON loss_sample (run_id, gross DESC);

CREATE TABLE ep_curve (run_id uuid NOT NULL REFERENCES scenario_run ON DELETE CASCADE, basis text NOT NULL CHECK (basis IN ('oep', 'aep')), perspective text NOT NULL CHECK (perspective IN ('gross', 'net')), return_period int NOT NULL, loss double precision NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (run_id, basis, perspective, return_period));
SELECT enable_tenant_rls('ep_curve');

CREATE TABLE damage_sample (run_id uuid NOT NULL REFERENCES scenario_run ON DELETE CASCADE, location_id uuid NOT NULL REFERENCES location, gust_ms real NOT NULL, surge_depth_m real NOT NULL, mean_damage_ratio real NOT NULL, gross_loss double precision NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (run_id, location_id));
SELECT enable_tenant_rls('damage_sample');
