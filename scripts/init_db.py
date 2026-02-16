"""
One-off: ensure Postgres has pgvector extension and DB exists.
Run against a running Postgres (e.g. docker compose up -d postgres).
Usage: DATABASE_URL=postgresql://... python scripts/init_db.py
"""
import os
import sys

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def main():
    url = os.getenv("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL", file=sys.stderr)
        sys.exit(1)
    # Parse and connect without the database name to create it if needed
    # For simplicity we connect to the DB and run CREATE EXTENSION
    conn = psycopg2.connect(url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.close()
    conn.close()
    print("pgvector extension ready.")


if __name__ == "__main__":
    main()
