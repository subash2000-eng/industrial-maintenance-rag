import os
import uuid
import json
from datetime import datetime
from pathlib  import Path
from sqlalchemy import create_engine, text
from dotenv   import load_dotenv

load_dotenv()

BASE_DIR       = Path(__file__).parent.parent
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "database/feedback.db"))


def get_engine():
    db_path = Path(SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_database():
    engine = get_engine()
    with engine.connect() as conn:
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                query_id    TEXT NOT NULL,
                rating      INTEGER,
                was_helpful INTEGER,
                comment     TEXT,
                timestamp   TEXT,
                FOREIGN KEY (query_id) REFERENCES query_log(query_id)
            )
        """))
        conn.commit()


def generate_query_id() -> str:
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
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT OR REPLACE INTO query_log
            (query_id, query, answer, confidence, chunks_used,
             response_time_s, model, sources_json, timestamp)
            VALUES
            (:query_id, :query, :answer, :confidence, :chunks_used,
             :response_time_s, :model, :sources_json, :timestamp)
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
    feedback_id = f"fb_{uuid.uuid4().hex[:12]}"
    engine      = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO feedback
            (feedback_id, query_id, rating, was_helpful, comment, timestamp)
            VALUES
            (:feedback_id, :query_id, :rating, :was_helpful, :comment, :timestamp)
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
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT query_id, query, confidence, chunks_used, response_time_s, timestamp
            FROM query_log
            ORDER BY timestamp DESC
            LIMIT :limit
        """), {"limit": limit})
        return [dict(row._mapping) for row in result]


def get_feedback_stats() -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) as total_feedback,
                   AVG(rating) as avg_rating,
                   SUM(was_helpful) as helpful_count
            FROM feedback
        """))
        row = result.fetchone()
        return dict(row._mapping) if row else {}


init_database()