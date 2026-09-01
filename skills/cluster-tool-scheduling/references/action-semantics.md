# Action Semantics

Use this reference for action legality, action-sequence validation, result validation, and scheduler code involving event boundaries.

All time intervals use `[start, end)` semantics unless the project states otherwise.

## Contents

- Core Occupancy Rules
- Pick
- Place
- Process / Align / Wait / Cool
- Clean
- Pump / Vent

## Core Occupancy Rules

- Chamber capacity: `Place.start` occupies the target chamber capacity; `Pick.end` releases the source chamber capacity.
- Process time: if a chamber step has process time, require `Pick.start - Place.end >= process_time`.
- Hand occupancy: `Pick.start` occupies the selected robot hand; the hand remains occupied by that wafer until the same wafer's `Place.end`.
- Cleaning occupancy: `Clean.start` occupies the PM for cleaning; `Clean.end` releases the PM.
- State transition: `Pump/Vent.start` starts an LL state transition; `Pump/Vent.end` completes the transition and updates the LL state.

## Pick

`Pick` takes a wafer from a source chamber into a robot hand.

Preconditions at `Pick.start`:

- The source chamber is currently occupied by the wafer.
- The wafer's process, align, cool, or wait time in the source chamber is complete.
- The robot can reach the source chamber.
- The source chamber state permits picking.
- The selected hand is empty.

Effects:

- From `Pick.start` to `Pick.end`, the robot, selected hand, and source chamber interface are in operation.
- The selected hand is occupied from `Pick.start` until the wafer's later `Place.end`.
- The source chamber capacity is released at `Pick.end`, not at `Pick.start`.

Timing:

```text
Pick.end - Pick.start = pick_time
```

## Place

`Place` puts a wafer from a robot hand into a target chamber.

Preconditions at `Place.start`:

- The selected hand holds the wafer.
- The target chamber is allowed by the wafer's current route step.
- The robot can reach the target chamber.
- The target chamber state permits placing.
- The target chamber has available capacity.

Effects:

- Target chamber capacity is occupied at `Place.start`.
- From `Place.start` to `Place.end`, the robot, selected hand, and target chamber interface are in operation.
- At `Place.end`, the wafer is in the target chamber and the hand is released.

Timing:

```text
Place.end - Place.start = place_time
```

## Process / Align / Wait / Cool

These actions represent a wafer staying in a chamber for required internal time.

If the wafer entered the chamber by `Place`, and leaves by `Pick`, require:

```text
Pick.start - Place.end >= required_internal_time
```

The chamber remains occupied for the full chamber occupancy interval:

```text
[Place.start, Pick.end)
```

## Clean

`Clean` is an exclusive PM action.

Preconditions at `Clean.start`:

- The PM is empty.
- The cleaning trigger is satisfied, unless the problem explicitly allows preventive cleaning.

Effects:

- The PM is occupied by cleaning on `[Clean.start, Clean.end)`.
- During cleaning, the PM cannot receive wafers, process wafers, be picked from, be placed into, or run another clean.

Timing:

```text
Clean.end = Clean.start + clean_time
```

If a clean must happen immediately after a wafer leaves a PM, and that wafer leaves by `Pick_prev`, require:

```text
Clean.start = Pick_prev.end
```

## Pump / Vent

`Pump` and `Vent` are state transitions for conversion LLs.

- `Pump`: atmosphere to vacuum.
- `Vent`: vacuum to atmosphere.

Preconditions at start:

- The LL is in the source state.
- No incompatible pick/place or other state transition is active for that LL during the transition interval.

Effects:

- During `[start, end)`, the LL is transitioning and cannot be used for pick/place.
- At `end`, the LL state becomes the target state.

Timing:

```text
end - start = transition_time
```

Important: describe Pump/Vent as state transitions, not as "occupying and releasing" the LL in the same way as wafer occupancy.
