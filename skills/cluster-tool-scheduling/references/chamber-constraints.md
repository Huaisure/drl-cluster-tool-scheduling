# Chamber Constraints

Use this reference for LP, AL, PM, LL, chamber occupancy, valves/interfaces, and residency constraints.

## Contents

- Chamber Occupancy Rule
- LP
- AL
- PM
- Conversion LL
- Pump / Vent State
- Conversion LL Slots and Direction
- Vacuum Transfer LL
- Candidate Modules
- Valves and Interfaces
- Residency

## Chamber Occupancy Rule

For every chamber:

- A wafer occupies the chamber from the entering `Place.start` to the leaving `Pick.end`.
- If chamber capacity is `C`, wafer occupancy count must never exceed `C`.
- If the chamber has internal process/align/cool/hold time, require:

```text
Pick.start - Place.end >= required_internal_time
```

## LP

LP is the cassette interface.

- Wafers start from LP slots and return to prescribed LP slots.
- Each slot has capacity 1.
- Leaving a slot by pick releases it at `Pick.end`.
- Returning to a slot by place occupies it at `Place.start`.

## AL

AL is the align chamber, usually capacity 1.

- Entering AL occupies it at `Place.start`.
- Leaving AL releases it at `Pick.end`.
- If alignment is required, require:

```text
Pick.start - Place.end >= align_time
```

## PM

PM is a process chamber, usually capacity 1.

- Entering PM occupies it at `Place.start`.
- Leaving PM releases it at `Pick.end`.
- If process time is `process_time`, require:

```text
Pick.start - Place.end >= process_time
```

While occupied by a wafer, the PM cannot receive another wafer or run cleaning.

## Conversion LL

Conversion LL connects atmosphere and vacuum.

- It has slot capacity.
- A wafer entering occupies an LL slot at `Place.start`.
- A wafer leaving releases the LL slot at `Pick.end`.
- The LL state must match the robot side at `Pick.start` and `Place.start`.

Common states:

- Atmosphere: atmospheric-side robot can pick/place.
- Vacuum: vacuum-side robot can pick/place.

## Pump / Vent State

Pump/Vent changes conversion LL state:

- `Pump`: atmosphere to vacuum.
- `Vent`: vacuum to atmosphere.

During `[start, end)`, the LL is transitioning and cannot be used for pick/place or another state transition. At `end`, the target state is active.

## Conversion LL Slots and Direction

If equipment defines slot directionality, place/pick must follow it. For example, a slot may be used only from atmosphere to vacuum or only from vacuum to atmosphere.

Each slot capacity is 1.

## Vacuum Transfer LL

Vacuum transfer LL is inside the vacuum region and does not require Pump/Vent.

It behaves like a chamber with capacity and optional hold/cool time:

```text
Pick.start - Place.end >= hold_time
```

If `hold_time = 0`, it is only a transfer node. It does not trigger PM cleaning.

## Candidate Modules

A route step may list multiple candidate modules. The wafer chooses one candidate for that step, and the chosen module follows normal chamber occupancy rules.

A wafer cannot occupy multiple candidates for the same route step.

## Valves and Interfaces

If modeled, valves/interfaces are resources.

- Pick/place occupies the corresponding interface during the pick/place action interval.
- The same interface cannot serve multiple actions simultaneously.
- If the equipment defines a group of mutually exclusive valves, those valve-open intervals must not overlap.

If explicit open/close actions are absent, treat the whole pick/place interval as occupying the corresponding interface.

## Residency

If the equipment limits post-process residency, compare completion time to the next relevant event.

Leaving-current-node limit:

```text
Pick.start - completion_time <= max_residency_time
```

Entering-next-node limit:

```text
next Place.end - completion_time <= max_transfer_time
```
