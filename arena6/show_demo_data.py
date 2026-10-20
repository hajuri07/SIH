"""
Arena 6 - Print the demo views of the seeded database (no SQL tool needed).

    python -m arena6.show_demo_data                # reads ./telemetry.db
    python -m arena6.show_demo_data --db my_demo.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = Path(__file__).resolve().parent / "sql" / "demo_queries.sql"


def load_queries() -> List[Tuple[str, str]]:
    """Parse demo_queries.sql into [(title, sql)]."""
    queries, title, buf = [], None, []
    for line in QUERY_FILE.read_text().splitlines():
        if line.startswith("-- name:"):
            if title and buf:
                queries.append((title, "\n".join(buf).strip()))
            title, buf = line[len("-- name:"):].strip(), []
        elif title is not None and not line.startswith("--"):
            buf.append(line)
    if title and buf:
        queries.append((title, "\n".join(buf).strip()))
    return queries


def format_table(headers, rows) -> str:
    cells = [[str(h) for h in headers]] + [["" if v is None else str(v) for v in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def run_all(conn: sqlite3.Connection) -> List[Tuple[str, list, list]]:
    results = []
    for title, sql in load_queries():
        cur = conn.execute(sql)
        results.append((title, [d[0] for d in cur.description], cur.fetchall()))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Show the Arena 6 demo views.")
    p.add_argument("--db", type=Path, default=ROOT / "telemetry.db")
    args = p.parse_args(argv)
    if not args.db.exists():
        print(f"ERROR: {args.db} not found. Run: python -m arena6.seed_db")
        return 2
    conn = sqlite3.connect(str(args.db))
    try:
        for title, headers, rows in run_all(conn):
            print(f"\n=== {title} ===")
            print(format_table(headers, rows) if rows else "(no rows)")
    except sqlite3.OperationalError as exc:
        print(f"ERROR: {exc}. Has the DB been seeded? Run: python -m arena6.seed_db")
        return 2
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
