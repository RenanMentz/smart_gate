"""Access history: capture storage and logging of access attempts.

Every attempt is stored in the ``access_logs`` table together with the image
that produced it, saved under ``data/captures/`` with a timestamped name
(``capture_20260101_093015_123456.jpg``).

Duplicate suppression
---------------------
A car standing in front of the camera is read many times. To keep the history
meaningful, a new record is **not** created when the most recent record of the
same plate has the same result *and* the same source (automatic / manual) and
is at most ``duplicate_window_seconds`` old. The existing record is returned
instead and the image is not saved again. A window of ``0`` disables the
suppression. Reads without a plate cannot be compared and are always logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

import config
from database.database import TIMESTAMP_FORMAT, AccessLog, Database

if TYPE_CHECKING:
    from modules.access_control import AccessDecision

logger = logging.getLogger(__name__)

JPEG_QUALITY = 90


@dataclass
class RecordOutcome:
    """Result of :meth:`AccessHistory.record`."""

    log: AccessLog
    duplicate: bool          # True if an existing record was returned instead of a new one


def stored_image_path(path: Path) -> str:
    """Path as stored in the database: relative to the project folder when possible.

    Relative paths keep the history valid if the project folder is moved.
    """
    try:
        return path.resolve().relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_image_path(stored: str | None) -> Path | None:
    """Inverse of :func:`stored_image_path`; ``None`` if there is no image or it is gone."""
    if not stored:
        return None
    path = Path(stored)
    if not path.is_absolute():
        path = config.BASE_DIR / path
    return path if path.is_file() else None


class AccessHistory:
    """Writes access attempts (and their images) to the database."""

    def __init__(
        self,
        db: Database,
        captures_dir: str | Path | None = None,
        duplicate_window_seconds: int | None = None,
    ) -> None:
        self.db = db
        self.captures_dir = Path(captures_dir) if captures_dir else config.CAPTURES_DIR
        self.duplicate_window_seconds = (
            config.DEFAULT_CONFIG.duplicate_window_seconds
            if duplicate_window_seconds is None else duplicate_window_seconds
        )

    # --------------------------------------------------------------- images
    def save_capture(self, image: np.ndarray, when: datetime | None = None) -> str:
        """Save ``image`` as JPEG in the captures folder and return the path to store.

        Raises:
            OSError: the image could not be encoded or written.
        """
        when = when or datetime.now()
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        path = self.captures_dir / f"capture_{when.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        try:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        except cv2.error as exc:
            raise OSError(f"The capture could not be encoded as JPEG: {exc}") from exc
        if not ok:
            raise OSError("The capture could not be encoded as JPEG.")
        path.write_bytes(encoded.tobytes())   # write_bytes: safe with non-ASCII Windows paths
        return stored_image_path(path)

    # ----------------------------------------------------------- duplicates
    def find_duplicate(self, plate: str | None, result, source: str, now: datetime) -> AccessLog | None:
        """Most recent record that makes a new one redundant, or ``None``."""
        if not plate or self.duplicate_window_seconds <= 0:
            return None
        last = self.db.last_log_for_plate(plate)
        if last is None or last.result != result or last.source != source:
            return None
        elapsed = (now - datetime.strptime(last.timestamp, TIMESTAMP_FORMAT)).total_seconds()
        return last if 0 <= elapsed <= self.duplicate_window_seconds else None

    # --------------------------------------------------------------- record
    def record(
        self,
        decision: AccessDecision,
        *,
        raw_text: str = "",
        confidence: float | None = None,
        detection_method: str = "",
        image: np.ndarray | None = None,
        image_path: str | None = None,
        source: str = "AUTO",
        notes: str = "",
        now: datetime | None = None,
    ) -> RecordOutcome:
        """Log an access attempt unless it is a duplicate.

        Args:
            decision: the access decision being recorded.
            raw_text / confidence / detection_method: OCR and detector data.
            image: captured image; saved when ``image_path`` is not given.
            image_path: path of an image that was already saved.
            source: ``"AUTO"`` (pipeline) or ``"MANUAL"`` (operator review).
            notes: free text stored with the record.
            now: current time (injectable for tests).
        """
        now = now or datetime.now()
        duplicate = self.find_duplicate(decision.plate, decision.result, source, now)
        if duplicate is not None:
            return RecordOutcome(duplicate, True)

        if image_path is None and image is not None:
            try:
                image_path = self.save_capture(image, now)
            except OSError:
                # the access attempt must be recorded even if the picture cannot be stored
                logger.exception("Could not save capture")
                notes = f"{notes} Capture could not be saved.".strip()

        log = self.db.add_access_log(
            decision.result,
            plate=decision.plate,
            raw_text=raw_text,
            confidence=confidence,
            detection_method=detection_method,
            source=source,
            image_path=image_path,
            vehicle_id=decision.vehicle.id if decision.vehicle else None,
            notes=notes,
            timestamp=now.strftime(TIMESTAMP_FORMAT),
        )
        return RecordOutcome(log, False)
