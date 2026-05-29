from sqlalchemy import text
from db_config_85 import get_engine

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS processing_route (
        route_id TEXT PRIMARY KEY,
        route_family TEXT,
        process_subtype TEXT,
        soak_hours DOUBLE PRECISION,
        ecap_angle_deg DOUBLE PRECISION
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS specimen (
        specimen_id TEXT PRIMARY KEY,
        route_id TEXT NOT NULL REFERENCES processing_route(route_id),
        sample_no INTEGER,
        ys_mpa DOUBLE PRECISION,
        uts_mpa DOUBLE PRECISION,
        elongation_percent DOUBLE PRECISION,
        hardness_hv DOUBLE PRECISION,
        grain_size_um DOUBLE PRECISION,
        cycles_to_failure DOUBLE PRECISION,
        tsa DOUBLE PRECISION,
        frequency_hz DOUBLE PRECISION,
        temperature_c DOUBLE PRECISION,
        is_synthetic BOOLEAN DEFAULT FALSE,
        source_file TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS specimen_processed (
        specimen_id TEXT PRIMARY KEY REFERENCES specimen(specimen_id),
        log_nf DOUBLE PRECISION,
        d_inv_sqrt DOUBLE PRECISION,
        strength_ratio DOUBLE PRECISION,
        fatigue_efficiency DOUBLE PRECISION,
        psa_mean DOUBLE PRECISION,
        psa_stable_mean DOUBLE PRECISION,
        psa_stable_std DOUBLE PRECISION,
        mean_stress_mean DOUBLE PRECISION,
        mean_stress_stable_mean DOUBLE PRECISION,
        stress_amp_mean DOUBLE PRECISION,
        stress_amp_stable_mean DOUBLE PRECISION,
        unloading_modulus_mean DOUBLE PRECISION,
        unloading_modulus_stable_mean DOUBLE PRECISION
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS cycle_summary (
        specimen_id TEXT NOT NULL REFERENCES specimen(specimen_id),
        route_id TEXT REFERENCES processing_route(route_id),
        cycle_no INTEGER NOT NULL,
        cycle_fraction DOUBLE PRECISION,
        psa DOUBLE PRECISION,
        mean_stress_mpa DOUBLE PRECISION,
        max_stress_mpa DOUBLE PRECISION,
        min_stress_mpa DOUBLE PRECISION,
        unloading_modulus_mpa DOUBLE PRECISION,
        stress_amplitude_mpa DOUBLE PRECISION,
        PRIMARY KEY (specimen_id, cycle_no)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS loop_manifest (
        specimen_id TEXT NOT NULL REFERENCES specimen(specimen_id),
        cycle_no INTEGER NOT NULL,
        file_path TEXT,
        n_rows INTEGER,
        issues TEXT,
        file_sha256 TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        PRIMARY KEY (specimen_id, cycle_no)
    );
    """,
]

# Keeps old v1 tables compatible if this script is run on an existing database.
ALTER_STATEMENTS = [
    "ALTER TABLE specimen ADD COLUMN IF NOT EXISTS sample_no INTEGER;",
    "ALTER TABLE specimen ADD COLUMN IF NOT EXISTS is_synthetic BOOLEAN DEFAULT FALSE;",
    "ALTER TABLE specimen ADD COLUMN IF NOT EXISTS source_file TEXT;",
    "ALTER TABLE cycle_summary ADD COLUMN IF NOT EXISTS route_id TEXT;",
    "ALTER TABLE cycle_summary ADD COLUMN IF NOT EXISTS cycle_fraction DOUBLE PRECISION;",
    "ALTER TABLE specimen_processed ADD COLUMN IF NOT EXISTS psa_stable_mean DOUBLE PRECISION;",
    "ALTER TABLE specimen_processed ADD COLUMN IF NOT EXISTS psa_stable_std DOUBLE PRECISION;",
    "ALTER TABLE specimen_processed ADD COLUMN IF NOT EXISTS mean_stress_stable_mean DOUBLE PRECISION;",
    "ALTER TABLE specimen_processed ADD COLUMN IF NOT EXISTS stress_amp_stable_mean DOUBLE PRECISION;",
    "ALTER TABLE specimen_processed ADD COLUMN IF NOT EXISTS unloading_modulus_stable_mean DOUBLE PRECISION;",
]

INDEX_AND_VIEW_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_specimen_route ON specimen(route_id);",
    "CREATE INDEX IF NOT EXISTS idx_cycle_specimen ON cycle_summary(specimen_id);",
    "CREATE INDEX IF NOT EXISTS idx_cycle_route ON cycle_summary(route_id);",
    "CREATE INDEX IF NOT EXISTS idx_manifest_specimen ON loop_manifest(specimen_id);",
    """
    CREATE OR REPLACE VIEW specimen_icme_view AS
    SELECT
        s.specimen_id,
        s.route_id,
        pr.route_family,
        pr.process_subtype,
        pr.soak_hours,
        pr.ecap_angle_deg,
        s.sample_no,
        s.ys_mpa,
        s.uts_mpa,
        s.elongation_percent,
        s.hardness_hv,
        s.grain_size_um,
        s.cycles_to_failure,
        s.tsa,
        s.frequency_hz,
        s.temperature_c,
        sp.log_nf,
        sp.d_inv_sqrt,
        sp.strength_ratio,
        sp.fatigue_efficiency,
        sp.psa_mean,
        sp.psa_stable_mean,
        sp.mean_stress_mean,
        sp.mean_stress_stable_mean,
        sp.stress_amp_mean,
        sp.stress_amp_stable_mean,
        sp.unloading_modulus_mean,
        sp.unloading_modulus_stable_mean
    FROM specimen s
    LEFT JOIN processing_route pr ON s.route_id = pr.route_id
    LEFT JOIN specimen_processed sp ON s.specimen_id = sp.specimen_id;
    """,
    """
    CREATE OR REPLACE VIEW v_route_sample_counts AS
    SELECT
        route_id,
        COUNT(*) AS n_samples,
        MIN(cycles_to_failure) AS min_nf,
        MAX(cycles_to_failure) AS max_nf,
        AVG(cycles_to_failure) AS mean_nf
    FROM specimen
    GROUP BY route_id
    ORDER BY route_id;
    """,
]


def main():
    print("=== START init_schema_85 ===")
    engine = get_engine()
    try:
        with engine.begin() as conn:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(text(stmt))
            for stmt in ALTER_STATEMENTS:
                conn.execute(text(stmt))
            for stmt in INDEX_AND_VIEW_STATEMENTS:
                conn.execute(text(stmt))
        print("✅ Schema created / verified successfully for the 85-sample workflow.")
    except Exception as exc:
        print("❌ Error while creating/verifying schema:")
        print(exc)


if __name__ == "__main__":
    main()
