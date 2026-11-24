"""SQLite persistence layer for Smart Gate.

Tables
------
``vehicles``     registered vehicles and their authorization status.
``access_logs``  every access attempt (automatic or manually reviewed).
``settings``     key/value pairs edited on the Settings screen.

Design notes
------------
* ``vehicles.plate`` is ``UNIQUE`` and always stored normalized
  (``ABC1D23``), which prevents duplicate registrations such as
  ``abc-1d23`` vs ``ABC1D23``.
* Deleting a vehicle keeps its history: ``access_logs.vehicle_id`` is set to
  ``NULL`` but the plate text recorded at the time is preserved.
* A new connection is opened per operation. SQLite connections are cheap and
  this avoids sharing a connection between Streamlit's threads.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from modules.plate_format import normalize_plate

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plate       TEXT    NOT NULL UNIQUE,
    model       TEXT    NOT NULL DEFAULT '',
    color       TEXT    NOT NULL DEFAULT '',
    owner       TEXT    NOT NULL DEFAULT '',
    authorized  INTEGER NOT NULL DEFAULT 1 CHECK (authorized IN (0, 1)),
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS access_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    plate             TEXT,                       -- normalized plate, NULL if unreadable
    raw_text          TEXT    NOT NULL DEFAULT '',-- text exactly as returned by the OCR
    result            TEXT    NOT NULL CHECK (result IN ('AUTHORIZED', 'DENIED', 'MANUAL_REVIEW')),
    confidence        REAL,                       -- OCR confidence 0..1, NULL if no OCR
    detection_method  TEXT    NOT NULL DEFAULT '',-- 'yolo', 'classical' or 'manual'
    source            TEXT    NOT NULL DEFAULT 'AUTO' CHECK (source IN ('AUTO', 'MANUAL')),
    image_path        TEXT,
    vehicle_id        INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
    notes             TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON access_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_plate     ON access_logs(plate);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class AccessResult(str, Enum):
    """The three possible outcomes of an access attempt."""

    AUTHORIZED = "AUTHORIZED"
    DENIED = "DENIED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class DatabaseError(Exception):
    """Base class for database errors raised by this module."""


class DuplicatePlateError(DatabaseError):
    """Raised when registering a plate that already exists."""


class VehicleNotFoundError(DatabaseError):
    """Raised when a vehicle id/plate does not exist."""


@dataclass
class Vehicle:
    """A registered vehicle."""

    id: int
    plate: str
    model: str
    color: str
    owner: str
    authorized: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Vehicle:
        data = dict(row)
        data["authorized"] = bool(data["authorized"])
        return cls(**data)


@dataclass
class AccessLog:
    """One access attempt."""

    id: int
    timestamp: str
    plate: str | None
    raw_text: str
    result: AccessResult
    confidence: float | None
    detection_method: str
    source: str
    image_path: str | None
    vehicle_id: int | None
    notes: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> AccessLog:
        data = dict(row)
        data["result"] = AccessResult(data["result"])
        return cls(**data)


def now_str() -> str:
    """Current local time in the format stored in the database."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


class Database:
    """Thin data-access object over the SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ core
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, roll back on error."""
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # -------------------------------------------------------------- vehicles
    def add_vehicle(
        self,
        plate: str,
        model: str = "",
        color: str = "",
        owner: str = "",
        authorized: bool = True,
    ) -> Vehicle:
        """Register a new vehicle.

        Raises:
            ValueError: invalid plate format.
            DuplicatePlateError: plate already registered.
        """
        plate = normalize_plate(plate)
        ts = now_str()
        try:
            with self.connect() as conn:
                cur = conn.execute(
                    "INSERT INTO vehicles (plate, model, color, owner, authorized, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (plate, model.strip(), color.strip(), owner.strip(), int(authorized), ts, ts),
                )
                vehicle_id = cur.lastrowid
        except sqlite3.IntegrityError as exc:
            raise DuplicatePlateError(f"Plate {plate} is already registered.") from exc
        return self.get_vehicle(vehicle_id)

    def get_vehicle(self, vehicle_id: int) -> Vehicle:
        """Return a vehicle by id or raise :class:`VehicleNotFoundError`."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM vehicles WHERE id = ?", (vehicle_id,)).fetchone()
        if row is None:
            raise VehicleNotFoundError(f"Vehicle id {vehicle_id} not found.")
        return Vehicle.from_row(row)

    def find_vehicle_by_plate(self, plate: str) -> Vehicle | None:
        """Return the vehicle with this plate, or ``None`` if not registered."""
        try:
            plate = normalize_plate(plate)
        except ValueError:
            return None
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM vehicles WHERE plate = ?", (plate,)).fetchone()
        return Vehicle.from_row(row) if row else None

    def update_vehicle(
        self,
        vehicle_id: int,
        *,
        plate: str | None = None,
        model: str | None = None,
        color: str | None = None,
        owner: str | None = None,
        authorized: bool | None = None,
    ) -> Vehicle:
        """Update the given fields of a vehicle (``None`` = keep current value)."""
        current = self.get_vehicle(vehicle_id)
        new_plate = normalize_plate(plate) if plate is not None else current.plate
        values = (
            new_plate,
            (model if model is not None else current.model).strip(),
            (color if color is not None else current.color).strip(),
            (owner if owner is not None else current.owner).strip(),
            int(authorized if authorized is not None else current.authorized),
            now_str(),
            vehicle_id,
        )
        try:
            with self.connect() as conn:
                conn.execute(
                    "UPDATE vehicles SET plate = ?, model = ?, color = ?, owner = ?,"
                    " authorized = ?, updated_at = ? WHERE id = ?",
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicatePlateError(f"Plate {new_plate} is already registered.") from exc
        return self.get_vehicle(vehicle_id)

    def set_authorization(self, vehicle_id: int, authorized: bool) -> Vehicle:
        """Authorize or block a vehicle."""
        return self.update_vehicle(vehicle_id, authorized=authorized)

    def delete_vehicle(self, vehicle_id: int) -> None:
        """Remove a vehicle. Its access history is kept (``vehicle_id`` -> NULL)."""
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,))
        if cur.rowcount == 0:
            raise VehicleNotFoundError(f"Vehicle id {vehicle_id} not found.")

    def list_vehicles(self, search: str = "", authorized: bool | None = None) -> list[Vehicle]:
        """List vehicles, optionally filtered by text and authorization status.

        ``search`` matches plate, model, color or owner (case-insensitive,
        hyphens/spaces in the plate are ignored).
        """
        query = "SELECT * FROM vehicles WHERE 1 = 1"
        params: list[Any] = []
        if search.strip():
            term = f"%{search.strip()}%"
            plate_term = f"%{''.join(ch for ch in search.upper() if ch.isalnum())}%"
            query += " AND (plate LIKE ? OR model LIKE ? OR color LIKE ? OR owner LIKE ?)"
            params += [plate_term, term, term, term]
        if authorized is not None:
            query += " AND authorized = ?"
            params.append(int(authorized))
        query += " ORDER BY plate"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [Vehicle.from_row(r) for r in rows]

    # ----------------------------------------------------------- access logs
    def add_access_log(
        self,
        result: AccessResult,
        *,
        plate: str | None = None,
        raw_text: str = "",
        confidence: float | None = None,
        detection_method: str = "",
        source: str = "AUTO",
        image_path: str | None = None,
        vehicle_id: int | None = None,
        notes: str = "",
        timestamp: str | None = None,
    ) -> AccessLog:
        """Insert an access attempt and return it."""
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO access_logs (timestamp, plate, raw_text, result, confidence,"
                " detection_method, source, image_path, vehicle_id, notes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp or now_str(),
                    plate,
                    raw_text,
                    AccessResult(result).value,
                    confidence,
                    detection_method,
                    source,
                    image_path,
                    vehicle_id,
                    notes,
                ),
            )
            row = conn.execute("SELECT * FROM access_logs WHERE id = ?", (cur.lastrowid,)).fetchone()
        return AccessLog.from_row(row)

    def list_access_logs(
        self,
        *,
        plate: str = "",
        result: AccessResult | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = 500,
    ) -> list[AccessLog]:
        """List access attempts, newest first.

        Args:
            plate: substring of the plate (normalization applied).
            result: filter by outcome.
            start / end: inclusive bounds as ``YYYY-MM-DD`` or full timestamps.
            limit: maximum number of rows (``None`` = no limit).
        """
        query = "SELECT * FROM access_logs WHERE 1 = 1"
        params: list[Any] = []
        if plate.strip():
            query += " AND plate LIKE ?"
            params.append(f"%{''.join(ch for ch in plate.upper() if ch.isalnum())}%")
        if result is not None:
            query += " AND result = ?"
            params.append(AccessResult(result).value)
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            # a bare date means "until the end of that day"
            query += " AND timestamp <= ?"
            params.append(end + " 23:59:59" if len(end) == 10 else end)
        query += " ORDER BY timestamp DESC, id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AccessLog.from_row(r) for r in rows]

    def last_log_for_plate(self, plate: str) -> AccessLog | None:
        """Most recent access attempt of a plate (used to avoid duplicate logs)."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM access_logs WHERE plate = ? ORDER BY timestamp DESC, id DESC LIMIT 1",
                (plate,),
            ).fetchone()
        return AccessLog.from_row(row) if row else None

    def count_logs_by_result(self, start: str | None = None) -> dict[AccessResult, int]:
        """Number of attempts per outcome, optionally since ``start``."""
        query = "SELECT result, COUNT(*) AS n FROM access_logs"
        params: list[Any] = []
        if start:
            query += " WHERE timestamp >= ?"
            params.append(start)
        query += " GROUP BY result"
        counts = {r: 0 for r in AccessResult}
        with self.connect() as conn:
            for row in conn.execute(query, params):
                counts[AccessResult(row["result"])] = row["n"]
        return counts

    # -------------------------------------------------------------- settings
    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def get_all_settings(self) -> dict[str, str]:
        with self.connect() as conn:
            return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings")}
