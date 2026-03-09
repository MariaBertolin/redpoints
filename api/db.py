import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path("analysis.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyzed_listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                is_asset INTEGER NOT NULL,
                stage1_score REAL NOT NULL,
                stage1_threshold REAL NOT NULL,
                similarity_score REAL NOT NULL,
                top_k INTEGER NOT NULL,
                top_matches_json TEXT NOT NULL,
                stage2_executed INTEGER NOT NULL,
                stage2_score REAL,
                stage2_threshold REAL NOT NULL,
                suspicion_flag INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_analysis(result: dict[str, Any]) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO analyzed_listings (
                title,
                is_asset,
                stage1_score,
                stage1_threshold,
                similarity_score,
                top_k,
                top_matches_json,
                stage2_executed,
                stage2_score,
                stage2_threshold,
                suspicion_flag,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["title"],
                int(result["is_asset"]),
                result["stage1"]["score"],
                result["stage1"]["threshold"],
                result["similarity"]["score"],
                result["similarity"]["top_k"],
                json.dumps(result["similarity"]["matches"], ensure_ascii=False),
                int(result["stage2"]["executed"]),
                result["stage2"]["score"],
                result["stage2"]["threshold"],
                (
                    None
                    if result["stage2"]["suspicion_flag"] is None
                    else int(result["stage2"]["suspicion_flag"])
                ),
                result["created_at"],
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_recent_listings(limit: int) -> list[sqlite3.Row]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM analyzed_listings
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows


def get_listings_by_threshold(stage: str, threshold: float) -> list[sqlite3.Row]:
    with get_connection() as conn:
        if stage == "stage1":
            rows = conn.execute(
                """
                SELECT *
                FROM analyzed_listings
                WHERE stage1_score >= ?
                ORDER BY id DESC
                """,
                (threshold,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM analyzed_listings
                WHERE is_asset = 1
                  AND stage2_executed = 1
                  AND stage2_score IS NOT NULL
                  AND stage2_score >= ?
                ORDER BY id DESC
                """,
                (threshold,),
            ).fetchall()
    return rows