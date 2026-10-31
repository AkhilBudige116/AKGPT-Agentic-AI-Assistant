"""Create the AKGPT SQLite schema (idempotent).

Usage:  python scripts/init_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from database import Database  # noqa: E402
from exceptions import ConfigurationError  # noqa: E402


def main() -> int:
    try:
        settings = get_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc.message}", file=sys.stderr)
        return 2
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(settings.database_url)
    db.init_schema()
    ok = db.ping()
    db.dispose()
    print(f"Database ready at {settings.sqlite_path} (ping={'ok' if ok else 'failed'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
