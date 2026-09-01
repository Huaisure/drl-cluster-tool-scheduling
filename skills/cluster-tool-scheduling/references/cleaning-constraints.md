# Cleaning Constraints

Use this reference for PM cleaning triggers, cleaning timing, replaying PM cleaning state, and cleaning conflict checks.

Cleaning applies to PMs. A clean is an exclusive PM action.

## Clean Action

At `Clean.start`:

- The PM must be empty.
- The cleaning trigger must be satisfied, unless preventive cleaning is explicitly allowed.

Timing:

```text
Clean.end = Clean.start + clean_time
```

During `[Clean.start, Clean.end)`:

- The PM cannot receive a wafer.
- The PM cannot process a wafer.
- The PM cannot be picked from or placed into.
- The PM cannot run another clean.

If a required clean must occur immediately after a wafer leaves, and the wafer leaves with `Pick_prev`, require:

```text
Clean.start = Pick_prev.end
```

## PM Cleaning State

Track per PM:

- Whether a wafer currently occupies the PM.
- Previous process type.
- Number of completed wafers since the last relevant clean.
- Idle time since the last wafer left or the last clean completed.
- Pending cleaning triggers.

## Type A: Idle-Time Clean

If PM idle time reaches the configured threshold, an idle clean is triggered.

If the previous wafer left at `Pick.end`, idle limit is `idle_limit`, and no wafer enters before the threshold:

```text
Clean.start = Pick.end + idle_limit
```

After the clean, idle time resets.

## Type B: Process-Switch Clean

If the next wafer's process type differs from the previous process type on the same PM, run process-switch cleaning before the next wafer enters.

If the previous wafer leaves with `Pick_prev` and the next wafer enters with `Place_next`, require:

```text
Clean.start = Pick_prev.end
Clean.end = Clean.start + clean_time
Clean.end <= Place_next.start
```

After the clean, update the PM process type to the next wafer's process type.

## Type C: Wafer-Count Clean

Each completed wafer increments the PM wafer count. When the count reaches the threshold, run wafer-count cleaning before the next wafer enters.

If the previous wafer leaves with `Pick_prev` and the next wafer enters with `Place_next`, require:

```text
Clean.start = Pick_prev.end
Clean.end = Clean.start + clean_time
Clean.end <= Place_next.start
```

After the clean, reset the wafer count to 0.

## Cleaning Priority

If multiple cleaning types trigger simultaneously, follow the problem-defined priority.

If the problem states that longer clean time has priority, execute the longer clean. If the problem states that process-switch clean has priority over wafer-count clean, execute process-switch clean first.

If no priority is defined, cleaning order is under-specified.
