"""Rebuild the FAISS index from the chunks stored in SQLite.

Run this after changing EMBEDDING_MODEL (vector dimensions / semantics change) or if the
index files were lost or corrupted. The chunk text is the source of truth, so no uploaded
file needs to be re-read.

Usage:  python scripts/rebuild_faiss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings  # noqa: E402
from database import Database  # noqa: E402
from exceptions import AKGPTError  # noqa: E402
from rag import RagService  # noqa: E402


def main() -> int:
    try:
        settings = get_settings()
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(settings.database_url)
        db.init_schema()
        rag = RagService(settings, db)
        rag.initialise()
        count = rag.rebuild()
    except AKGPTError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1
    print(f"Rebuilt FAISS index with {count} vectors at {settings.faiss_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
