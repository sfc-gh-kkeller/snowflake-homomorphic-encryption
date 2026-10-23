-- Demo tables. Run by `pixi run deploy` (he_snowflake/deploy.py) in the configured DATABASE.SCHEMA.
-- Plaintext columns are deliberately non-sensitive dimensions (they are what Snowflake filters/groups on).

CREATE OR REPLACE TABLE HE_KEYS (
    key_id         STRING PRIMARY KEY,
    scheme         STRING,              -- bfv | ckks
    mode           STRING,              -- row | packed
    params         VARIANT,
    context        BINARY,              -- public SEAL encryption parameters only (~100 bytes)
    rotation_keys  BINARY(67108864),    -- packed mode only: public Galois keys (no secret material)
    created_at     TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE EMPLOYEES_ENC (
    emp_id          INT,
    department      STRING,
    region          STRING,
    hire_year       INT,
    fte_pct         INT,
    bfv_key_id      STRING,
    salary_ct       BINARY,   -- BFV: annual salary (USD)
    bonus_ct        BINARY,   -- BFV: annual bonus (USD)
    deductions_ct   BINARY,   -- BFV: annual deductions (USD)
    ckks_key_id     STRING,
    hourly_rate_ct  BINARY,   -- CKKS: contractor-style hourly rate
    hours_ct        BINARY,   -- CKKS: hours logged this month
    perf_score_ct   BINARY,   -- CKKS: performance score 1..5
    tenure_ct       BINARY,   -- CKKS: tenure in years
    remote_flag_ct  BINARY    -- CKKS: 1 = remote, 0 = office (sensitive, so encrypted)
);

CREATE OR REPLACE TABLE REVENUE_MONTHLY_ENC (
    region       STRING,
    month        DATE,
    bfv_key_id   STRING,
    revenue_ct   BINARY       -- BFV: monthly revenue (USD)
);

CREATE OR REPLACE TABLE PACKED_COLUMNS (
    table_name   STRING,
    column_name  STRING,
    key_id       STRING,
    n_values     INT,          -- values packed into slots 0..n_values-1, ordered by emp_id
    ct           BINARY        -- the entire column in ONE ciphertext
);

CREATE OR REPLACE TABLE FX_RATES (
    region        STRING,
    usd_to_local  FLOAT
);
