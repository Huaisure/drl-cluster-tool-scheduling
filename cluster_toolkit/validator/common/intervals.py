from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ..models import TimeValue


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class Interval(Generic[PayloadT]):
    start: TimeValue
    end: TimeValue
    payload: PayloadT | None = None


def intervals_overlap(
    left: Interval[object],
    right: Interval[object],
    *,
    epsilon: float = 0.0,
) -> bool:
    """Return whether two half-open intervals overlap by more than epsilon."""

    return left.start < right.end - epsilon and right.start < left.end - epsilon


def within_closed_window(
    value: TimeValue,
    lower: TimeValue,
    upper: TimeValue,
    *,
    epsilon: float = 0.0,
) -> bool:
    """Return whether ``value`` is inside the inclusive window ``[lower, upper]``."""

    return lower - epsilon <= value <= upper + epsilon
