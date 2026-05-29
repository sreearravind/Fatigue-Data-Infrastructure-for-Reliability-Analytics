from sqlalchemy import text
from db_config_85 import get_engine, DB_NAME

EXPECTED_TABLES = [
    "processing_route",
    "specimen",
    "specimen_processed",
    "cycle_summary",
    "loop_manifest",
]


def table_exists(conn, table_name: str) -> bool:
    return bool(conn.execute(text("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = :table_name
        );
    """), {"table_name": table_name}).scalar())


def main():
    print("=== START db_connection_test_85 ===")
    engine = get_engine()

    try:
        with engine.connect() as conn:
            current_db = conn.execute(text("SELECT current_database();")).scalar()
            version = conn.execute(text("SELECT version();")).scalar()
            print(f"✅ Connected to database: {current_db}")
            print(f"PostgreSQL version: {version.split(',')[0]}")

            if current_db != DB_NAME:
                print(f"⚠️ Connected DB differs from configured DB_NAME: {DB_NAME}")

            print("\nTable availability and counts:")
            for table in EXPECTED_TABLES:
                if table_exists(conn, table):
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table};")).scalar()
                    print(f"  ✅ {table:<20} rows = {count}")
                else:
                    print(f"  ⚠️ {table:<20} not created yet")

        print("\n✅ DB connection test completed.")
    except Exception as exc:
        print("❌ DB connection failed.")
        print("Check that PostgreSQL is running, the database exists, and DB credentials are correct in db_config_85.py")
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
