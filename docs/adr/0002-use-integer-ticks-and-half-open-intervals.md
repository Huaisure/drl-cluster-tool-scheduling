---
status: accepted
---

# Use integer ticks and half-open intervals

Use exact integer ticks inside the compiler, kernel, schedule, and validator, while accepting physical durations such as seconds at the external boundary. All occupied intervals use `[start, end)`, and a deadline is satisfied when `satisfy_time <= deadline`. This avoids floating-point ambiguity and makes adjacent operations at the same timestamp non-overlapping by definition.

## Consequences

Input compilation must declare a time unit and reject values that cannot be represented at the selected precision. All events at one tick are resolved atomically to a deterministic fixed point before the next decision epoch.
