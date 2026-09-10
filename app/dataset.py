import csv
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are",
    "was", "were", "what", "who", "how", "why", "when", "where", "which", "does",
    "do", "did", "with", "about", "tell", "me", "give", "show", "explain",
}


@dataclass
class Doc:
    doc_id: str
    title: str
    year: str
    plot: str
    score: float


def _ensure_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
    except sqlite3.OperationalError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "This Python's sqlite3 was built without FTS5. Use a python.org build "
            "or a conda/homebrew Python that bundles FTS5."
        ) from exc


def build_index(csv_path: str | Path, db_path: str | Path) -> int:
    """(Re)build the FTS5 index from the CSV. Returns the row count."""
    csv_path, db_path = Path(csv_path), Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        _ensure_fts5(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE movies USING fts5("
            "doc_id UNINDEXED, title, year UNINDEXED, plot, tokenize='porter unicode61')"
        )
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = [
                (r["doc_id"], r["title"], r["year"], r["plot"])
                for r in csv.DictReader(fh)
            ]
        conn.executemany("INSERT INTO movies VALUES (?, ?, ?, ?)", rows)
        conn.commit()
        logger.info("built FTS5 index at %s with %d rows", db_path, len(rows))
        return len(rows)
    finally:
        conn.close()


def ensure_index(csv_path: str | Path, db_path: str | Path) -> None:
    if not Path(db_path).exists():
        build_index(csv_path, db_path)


def _match_query(question: str) -> str:
    tokens = [t.lower() for t in _TOKEN_RE.findall(question)]
    tokens = [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]
    # Quote each term so FTS5 treats it as a literal, OR them for recall.
    return " OR ".join(f'"{t}"' for t in tokens)


class DatasetSearch:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    def search(self, question: str, k: int = 3) -> list[Doc]:
        query = _match_query(question)
        if not query:
            return []
        conn = sqlite3.connect(self._db_path)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT doc_id, title, year, plot, bm25(movies) AS score "
                "FROM movies WHERE movies MATCH ? ORDER BY score LIMIT ?",
                (query, k),
            )
            return [
                Doc(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    year=row["year"],
                    plot=row["plot"],
                    score=float(row["score"]),
                )
                for row in cur.fetchall()
            ]
        finally:
            conn.close()
