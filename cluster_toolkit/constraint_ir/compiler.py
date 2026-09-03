from __future__ import annotations

from decimal import Decimal, InvalidOperation

from .diagnostics import DiagnosticCode, SemanticError
from .schema import TimeDomain


def compile_ticks(
    time_domain: TimeDomain,
    value: Decimal | int | float | str,
    *,
    path: str,
) -> int:
    """Convert one external time value to exact non-negative integer ticks."""

    if isinstance(value, bool):
        raise _invalid_time(value, path)
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise _invalid_time(value, path) from None

    if not decimal_value.is_finite() or decimal_value < 0:
        raise _invalid_time(value, path)

    scaled = decimal_value * time_domain.ticks_per_unit
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise SemanticError(
            DiagnosticCode.TIME_PRECISION_LOSS,
            (
                f"{path}={value!r} {time_domain.unit} cannot be represented "
                f"exactly with {time_domain.ticks_per_unit} ticks per unit"
            ),
            path=path,
            details={"value": str(value), "scaled_ticks": str(scaled)},
        )
    return int(integral)


def _invalid_time(value: object, path: str) -> SemanticError:
    return SemanticError(
        DiagnosticCode.INVALID_TIME_VALUE,
        f"{path} must be a finite non-negative time value, got {value!r}",
        path=path,
    )
