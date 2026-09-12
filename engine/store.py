"""SQLite persistence for enrolled people and presence events.

SQLite rather than a CSV file because presence is a question with a shape:
who is enrolled, when did they arrive, were they here yesterday. A flat file
answers none of those without being parsed back into a database anyway, and it
has no way to express that the same person passing the camera twice in an hour
is one arrival rather than two.

Standard library only - no ORM, no migration framework. The schema is small
enough to read in one sitting.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import DB_PATH, DEDUPE_WINDOW_HOURS

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);

-- One row per reference image. Multiple rows per person is the point: each
-- covers a different pose or lighting condition.
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    source      TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);

-- A presence record, written only after a face clears every gate including
-- the liveness challenge.
CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY,
    person_id         INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    at                TEXT NOT NULL,
    confidence        REAL NOT NULL,
    similarity        REAL NOT NULL,
    challenges_passed INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);
CREATE INDEX IF NOT EXISTS idx_events_person_at ON events(person_id, at);
"""


def _now() -> str:
    """Timestamps are stored UTC and ISO-formatted so they sort lexically."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PresenceEvent:
    name: str
    at: str
    confidence: float
    similarity: float
    challenges_passed: int


class Store:
    """Owns the database connection and every statement run against it."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def _write(self):
        with self._connection:
            yield self._connection

    # --- enrolment -------------------------------------------------------

    def add_person(self, name: str) -> int:
        with self._write() as c:
            c.execute(
                "INSERT OR IGNORE INTO people (name, created_at) VALUES (?, ?)",
                (name, _now()),
            )
        row = self._connection.execute(
            "SELECT id FROM people WHERE name = ?", (name,)
        ).fetchone()
        return int(row["id"])

    def add_embedding(self, name: str, vector: np.ndarray, source: str = "") -> None:
        person_id = self.add_person(name)
        with self._write() as c:
            c.execute(
                "INSERT INTO embeddings (person_id, vector, source, created_at)"
                " VALUES (?, ?, ?, ?)",
                (person_id, np.asarray(vector, dtype=np.float32).tobytes(), source, _now()),
            )

    def delete_person(self, name: str) -> None:
        with self._write() as c:
            c.execute("DELETE FROM people WHERE name = ?", (name,))

    def load_embeddings(self) -> dict[str, list[np.ndarray]]:
        """Every enrolled person and their reference vectors."""
        rows = self._connection.execute(
            "SELECT p.name, e.vector FROM embeddings e"
            " JOIN people p ON p.id = e.person_id ORDER BY p.name"
        ).fetchall()

        people: dict[str, list[np.ndarray]] = {}
        for row in rows:
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            people.setdefault(row["name"], []).append(vector)
        return people

    def roster(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT p.name, p.created_at, COUNT(e.id) AS references_held,"
            " (SELECT MAX(at) FROM events WHERE person_id = p.id) AS last_seen"
            " FROM people p LEFT JOIN embeddings e ON e.person_id = p.id"
            " GROUP BY p.id ORDER BY p.name"
        ).fetchall()
        return [dict(row) for row in rows]

    # --- presence --------------------------------------------------------

    def recently_logged(self, name: str, within_hours: float = DEDUPE_WINDOW_HOURS) -> bool:
        """Whether this person already has a presence record in the window."""
        # A window of zero means no suppression at all. Without this the
        # comparison below is inclusive of the current second, so a zero-length
        # window would still swallow a log written moments earlier.
        if within_hours <= 0:
            return False

        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=within_hours)
        ).isoformat(timespec="seconds")
        row = self._connection.execute(
            "SELECT 1 FROM events e JOIN people p ON p.id = e.person_id"
            " WHERE p.name = ? AND e.at >= ? LIMIT 1",
            (name, cutoff),
        ).fetchone()
        return row is not None

    def log_presence(
        self,
        name: str,
        confidence: float,
        similarity: float,
        challenges_passed: int,
        within_hours: float = DEDUPE_WINDOW_HOURS,
    ) -> bool:
        """Record a presence unless one already stands inside the window.

        Returns whether a row was actually written, so the caller can tell a
        fresh arrival from a repeat sighting.
        """
        if self.recently_logged(name, within_hours):
            return False

        person_id = self.add_person(name)
        with self._write() as c:
            c.execute(
                "INSERT INTO events (person_id, at, confidence, similarity,"
                " challenges_passed) VALUES (?, ?, ?, ?, ?)",
                (person_id, _now(), confidence, similarity, challenges_passed),
            )
        return True

    def recent_events(self, limit: int = 50) -> list[PresenceEvent]:
        rows = self._connection.execute(
            "SELECT p.name, e.at, e.confidence, e.similarity, e.challenges_passed"
            " FROM events e JOIN people p ON p.id = e.person_id"
            " ORDER BY e.at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            PresenceEvent(
                name=r["name"],
                at=r["at"],
                confidence=r["confidence"],
                similarity=r["similarity"],
                challenges_passed=r["challenges_passed"],
            )
            for r in rows
        ]

    def present_on(self, day: str) -> list[str]:
        """Names recorded present on a given UTC date (YYYY-MM-DD)."""
        rows = self._connection.execute(
            "SELECT DISTINCT p.name FROM events e JOIN people p ON p.id = e.person_id"
            " WHERE substr(e.at, 1, 10) = ? ORDER BY p.name",
            (day,),
        ).fetchall()
        return [r["name"] for r in rows]
