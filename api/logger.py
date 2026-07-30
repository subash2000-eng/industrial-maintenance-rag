"""
Phase 4 - Query Logger
Logs all queries, responses, and feedback to SQLite.
Demonstrates production-grade observability.
"""

import os
import uuid
import json
from datetime import datetime
from pathlib  import Path
from sqlalchemy import create_engine, text
from dotenv   import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.parent
SQLITE_DB_PATH = os.getenv(
    "SQLITE_DB_PATH",
    str(BASE_DIR / "database/feedback.db")
)


def get_engine():
    """
    Create SQLAlchemy engine for SQLite.
    SQLite stores everything in one .db file — no server needed.
    """
    db_path = Path(SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        f"sqlite:///{db_path}",
        echo=False
    )


def init_database():
    """
    Create tables if they do not exist yet.
    Safe to call multiple times — only creates if missing.
    """
    engine = get_engine()

    with engine.connect() as conn:

        # ── Query log table ──────────────────────────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_log (
                query_id        TEXT PRIMARY KEY,
                query           TEXT NOT NULL,
                answer          TEXT,
                confidence      TEXT,
                chunks_used     INTEGER,
                response_time_s REAL,
                model           TEXT,
                sources_json    TEXT,
                timestamp       TEXT
            )
        """))

        # ── Feedback table ───────────────────────────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                query_id    TEXT NOT NULL,
                rating      INTEGER,
                was_helpful INTEGER,
                comment     TEXT,
                timestamp   TEXT,
                FOREIGN KEY (query_id)
                    REFERENCES query_log(query_id)
            )
        """))

        conn.commit()

    print(f"  ✅ Database initialized: {SQLITE_DB_PATH}")


def generate_query_id() -> str:
    """
    Generate a unique ID for each query.
    Format: q_ + 12 random hex characters
    Example: q_3f8a2c9d1b4e
    """
    return f"q_{uuid.uuid4().hex[:12]}"


def log_query(
    query_id       : str,
    query          : str,
    answer         : str,
    confidence     : str,
    chunks_used    : int,
    response_time_s: float,
    model          : str,
    sources        : list
) -> None:
    """
    Save a query and its response to the database.

    Called automatically in the background after
    every successful query — does not slow down response.
    """
    engine = get_engine()

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO query_log
            (
                query_id, query, answer, confidence,
                chunks_used, response_time_s, model,
                sources_json, timestamp
            )
            VALUES
            (
                :query_id, :query, :answer, :confidence,
                :chunks_used, :response_time_s, :model,
                :sources_json, :timestamp
            )
        """), {
            "query_id"       : query_id,
            "query"          : query,
            "answer"         : answer,
            "confidence"     : confidence,
            "chunks_used"    : chunks_used,
            "response_time_s": response_time_s,
            "model"          : model,
            "sources_json"   : json.dumps(sources),
            "timestamp"      : datetime.now().isoformat()
        })
        conn.commit()


def log_feedback(
    query_id   : str,
    rating     : int,
    was_helpful: bool,
    comment    : str = None
) -> str:
    """
    Save technician feedback for a query.
    Returns the feedback_id.
    """
    feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
    engine      = get_engine()

    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO feedback
            (
                feedback_id, query_id, rating,
                was_helpful, comment, timestamp
            )
            VALUES
            (
                :feedback_id, :query_id, :rating,
                :was_helpful, :comment, :timestamp
            )
        """), {
            "feedback_id": feedback_id,
            "query_id"   : query_id,
            "rating"     : rating,
            "was_helpful": int(was_helpful),
            "comment"    : comment,
            "timestamp"  : datetime.now().isoformat()
        })
        conn.commit()

    return feedback_id


def get_recent_queries(limit: int = 10) -> list:
    """
    Fetch most recent queries from the log.
    Used by the /history API endpoint.
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                query_id,
                query,
                confidence,
                chunks_used,
                response_time_s,
                timestamp
            FROM query_log
            ORDER BY timestamp DESC
            LIMIT :limit
        """), {"limit": limit})

        return [
            dict(row._mapping)
            for row in result
        ]


def get_feedback_stats() -> dict:
    """
    Get overall feedback statistics.
    Shows average rating and helpful percentage.
    """
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                COUNT(*)       as total_feedback,
                AVG(rating)    as avg_rating,
                SUM(was_helpful) as helpful_count
            FROM feedback
        """))
        row = result.fetchone()
        return dict(row._mapping) if row else {}


# ── Initialize database when this module is imported ────
init_database()


# ── Run this file directly to test ──────────────────────
if __name__ == "__main__":

    print("🧪 Testing Query Logger...\n")

    # Test 1: Log a sample query
    print("📝 Logging sample query...")
    test_id = generate_query_id()
    print(f"   Query ID: {test_id}")

    log_query(
        query_id        =test_id,
        query           ="motor bearing overheating fault",
        answer          ="## Fault Diagnosis\nBearing overheating...",
        confidence      ="High",
        chunks_used     =3,
        response_time_s =28.4,
        model           ="mistral",
        sources         =[{
            "source_num"   : 1,
            "manual_name"  : "SIEMENS Motor Manual",
            "page_number"  : 16,
            "section_title": "Troubleshooting",
            "confidence"   : "High",
            "rerank_score" : 6.82
        }]
    )
    print(f"  ✅ Query logged successfully")

    # Test 2: Log feedback for that query
    print(f"\n⭐ Logging feedback...")
    fb_id = log_feedback(
        query_id   =test_id,
        rating     =4,
        was_helpful=True,
        comment    ="Steps were accurate and clear"
    )
    print(f"   Feedback ID: {fb_id}")
    print(f"  ✅ Feedback logged successfully")

    # Test 3: Retrieve recent queries
    print(f"\n📋 Recent queries:")
    recent = get_recent_queries(limit=5)
    for q in recent:
        print(
            f"   {q['query_id']} | "
            f"{q['confidence']} | "
            f"{q['response_time_s']}s | "
            f"{q['query'][:40]}..."
        )

    # Test 4: Get feedback stats
    print(f"\n📊 Feedback statistics:")
    stats = get_feedback_stats()
    print(f"   Total feedback : {stats.get('total_feedback', 0)}")
    print(f"   Average rating : {stats.get('avg_rating', 0):.1f}/5")
    print(f"   Helpful count  : {stats.get('helpful_count', 0)}")

    print(f"\n✅ Logger working correctly")
    print(f"   Database saved at: {SQLITE_DB_PATH}")