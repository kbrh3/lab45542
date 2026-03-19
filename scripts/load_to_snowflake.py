"""
scripts/load_to_snowflake.py

Loads legislative bill data from CSV into Snowflake POLICYPULSE_DB.PUBLIC.BILLS.
Expects a CSV with columns matching the BILLS table schema.

Usage:
    python scripts/load_to_snowflake.py --csv data/bills_export.csv
    python scripts/load_to_snowflake.py --csv data/bills_export.csv --truncate
"""

import argparse
import csv
import os
import sys

try:
    import snowflake.connector
except ImportError:
    print("snowflake-connector-python is required. Install: pip install snowflake-connector-python")
    sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV = [
    "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD",
    "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA",
]


def get_connection():
    missing = [v for v in REQUIRED_ENV if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
    )


INSERT_SQL = """
INSERT INTO POLICYPULSE_DB.PUBLIC.BILLS
    (BILL_ID, SESSION_ID, BILL_NUMBER, STATUS, STATUS_DESC, STATUS_DATE,
     TITLE, DESCRIPTION, COMMITTEE, COMMITTEE_ID, LAST_ACTION, LAST_ACTION_DATE,
     URL, STATE_LINK, CHANGE_HASH)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def load_csv(csv_path: str, truncate: bool = False):
    conn = get_connection()
    cur = conn.cursor()

    try:
        if truncate:
            cur.execute("TRUNCATE TABLE POLICYPULSE_DB.PUBLIC.BILLS")
            print("Truncated BILLS table.")

        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((
                    int(row.get("BILL_ID", 0)),
                    int(row.get("SESSION_ID", 0)) if row.get("SESSION_ID") else None,
                    row.get("BILL_NUMBER", ""),
                    int(row.get("STATUS", 0)) if row.get("STATUS") else None,
                    row.get("STATUS_DESC", ""),
                    row.get("STATUS_DATE") or None,
                    row.get("TITLE", ""),
                    row.get("DESCRIPTION", ""),
                    row.get("COMMITTEE", ""),
                    int(row.get("COMMITTEE_ID", 0)) if row.get("COMMITTEE_ID") else None,
                    row.get("LAST_ACTION", ""),
                    row.get("LAST_ACTION_DATE") or None,
                    row.get("URL", ""),
                    row.get("STATE_LINK", ""),
                    row.get("CHANGE_HASH", ""),
                ))

        if not rows:
            print("No rows found in CSV.")
            return

        BATCH = 500
        total = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            cur.executemany(INSERT_SQL, batch)
            total += len(batch)
            print(f"  Inserted {total}/{len(rows)} rows...")

        print(f"Done. Loaded {total} rows into BILLS.")

        cur.execute("SELECT COUNT(*) FROM POLICYPULSE_DB.PUBLIC.BILLS")
        count = cur.fetchone()[0]
        print(f"Verification: BILLS table now has {count} rows.")

    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Load bill data CSV into Snowflake")
    parser.add_argument("--csv", required=True, help="Path to bills CSV file")
    parser.add_argument("--truncate", action="store_true", help="Truncate table before loading")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}")
        sys.exit(1)

    load_csv(args.csv, truncate=args.truncate)


if __name__ == "__main__":
    main()
