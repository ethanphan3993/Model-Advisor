"""Shared pytest fixtures.

Each test gets a fresh isolated SQLite DB so we never touch the user's real cache.
"""

import os
import tempfile
from pathlib import Path

import pytest

# Redirect DB to a temp file BEFORE importing modules that read DB_PATH at import time.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="model_advisor_test_"))
os.environ["MODEL_ADVISOR_TEST_DB"] = str(_TMP_DIR / "test.db")

import backend.db as db
db.DB_PATH = Path(os.environ["MODEL_ADVISOR_TEST_DB"])


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Each test gets a fresh DB and a clean recommender cache."""
    p = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", p)
    db.init_db()
    from backend.services import recommender
    recommender.clear_cache()
    yield
