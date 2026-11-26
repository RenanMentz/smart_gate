"""Access control: decide whether a vehicle may enter.

Decision rules (:func:`decide_access`), evaluated in this order:

1. The OCR asked for review (``needs_review``) or there is no plate
   -> ``MANUAL_REVIEW``. A doubtful read is never allowed to open the gate.
2. The plate is registered and authorized -> ``AUTHORIZED``.
3. The plate is registered but blocked -> ``DENIED``.
4. The plate is not registered:

   * if a plate that differs only by a confusable character at position 5
     (``ABC1B23`` <-> ``ABC1823``, see :func:`modules.ocr.position4_alternatives`)
     *is* registered -> ``MANUAL_REVIEW``. The OCR cannot know whether that
     character is a letter or a digit, so the read may be a registered
     vehicle and an operator must look at the picture;
   * otherwise -> ``DENIED``.

Manual review
-------------
When an operator confirms or types a plate, the human has already resolved the
doubt, so :meth:`AccessController.review_manually` decides again with
``allow_review=False``: the answer is always ``AUTHORIZED`` or ``DENIED`` and
never goes back to the review queue. The new record is stored with
``source="MANUAL"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from database.database import AccessLog, AccessResult, Database, Vehicle
from modules.ocr import OCRResult, position4_alternatives
from modules.plate_format import normalize_plate

if TYPE_CHECKING:  # avoids a runtime import cycle; history only needs the decision type
    from modules.history import AccessHistory, RecordOutcome


@dataclass
class AccessDecision:
    """Outcome of the access rules for one read."""

    result: AccessResult
    plate: str | None                       # normalized plate that was looked up (None if unreadable)
    reason: str                             # one sentence explaining the decision (for the operator)
    vehicle: Vehicle | None = None          # registered vehicle that matched the plate, if any
    similar_vehicles: list[Vehicle] = field(default_factory=list)  # registered "twin" plates

    @property
    def opens_gate(self) -> bool:
        return self.result is AccessResult.AUTHORIZED


def decide_access(
    db: Database,
    plate: str | None,
    *,
    needs_review: bool = False,
    review_reasons: Sequence[str] = (),
    allow_review: bool = True,
) -> AccessDecision:
    """Apply the access rules to a plate.

    Args:
        db: database holding the registered vehicles.
        plate: plate returned by the OCR (``None`` if nothing valid was read).
        needs_review: the OCR flagged this read as doubtful.
        review_reasons: the OCR's reasons, copied into the explanation.
        allow_review: set to ``False`` for plates confirmed by an operator;
            doubt flags and look-alike plates are then ignored.
    """
    if allow_review and (needs_review or not plate):
        detail = " ".join(review_reasons) if review_reasons else "No valid plate was read."
        return AccessDecision(AccessResult.MANUAL_REVIEW, plate, f"Manual review required. {detail}".strip())
    if not plate:  # an operator must always provide a plate
        raise ValueError("A plate is required for a confirmed decision.")

    vehicle = db.find_vehicle_by_plate(plate)
    if vehicle is not None:
        if vehicle.authorized:
            return AccessDecision(AccessResult.AUTHORIZED, plate, "Registered and authorized vehicle.", vehicle)
        return AccessDecision(AccessResult.DENIED, plate, "Registered vehicle is blocked.", vehicle)

    if allow_review:
        similar = [v for alt in position4_alternatives(plate)
                   if (v := db.find_vehicle_by_plate(alt)) is not None]
        if similar:
            twins = ", ".join(v.plate for v in similar)
            return AccessDecision(
                AccessResult.MANUAL_REVIEW, plate,
                f"Plate not registered, but a similar registered plate exists ({twins}): "
                "the 5th character may have been misread.",
                similar_vehicles=similar,
            )
    return AccessDecision(AccessResult.DENIED, plate, "Plate is not registered.")


@dataclass
class ManualReviewOutcome:
    """Result of an operator's manual review."""

    decision: AccessDecision
    record: RecordOutcome


class AccessController:
    """Applies the access rules and records manual reviews."""

    def __init__(self, db: Database, history: AccessHistory) -> None:
        self.db = db
        self.history = history

    def evaluate(self, ocr: OCRResult | None, error: str | None = None) -> AccessDecision:
        """Decision for an OCR result (``ocr=None`` means the OCR could not run)."""
        if ocr is None:
            reason = error or "OCR did not run."
            return decide_access(self.db, None, needs_review=True, review_reasons=[reason])
        return decide_access(self.db, ocr.plate, needs_review=ocr.needs_review, review_reasons=ocr.reasons)

    def review_manually(
        self,
        plate_text: str,
        *,
        image: np.ndarray | None = None,
        image_path: str | None = None,
        raw_text: str = "",
        confidence: float | None = None,
        detection_method: str = "",
        original_log: AccessLog | None = None,
    ) -> ManualReviewOutcome:
        """Decide again using a plate typed or confirmed by the operator.

        Args:
            plate_text: plate as typed by the operator (any case, with or without hyphen).
            image: captured image, saved only if ``image_path`` is not given.
            image_path: image already stored by the automatic attempt.
            raw_text / confidence / detection_method: values of the automatic read,
                kept for traceability (``detection_method`` falls back to ``"manual"``).
            original_log: the ``MANUAL_REVIEW`` record being resolved; referenced in the notes.

        Raises:
            ValueError: ``plate_text`` is not a valid Brazilian plate.
        """
        plate = normalize_plate(plate_text)
        decision = decide_access(self.db, plate, allow_review=False)
        notes = "Resolved by operator."
        if original_log is not None:
            notes = f"Resolved by operator (review of log #{original_log.id})."
        record = self.history.record(
            decision,
            raw_text=raw_text,
            confidence=confidence,
            detection_method=detection_method or "manual",
            image=image,
            image_path=image_path,
            source="MANUAL",
            notes=notes,
        )
        return ManualReviewOutcome(decision=decision, record=record)
