# Robot Constraints

Use this reference for TM reachability, movement, pick/place timing, hand occupancy, dual-arm geometry, and robot conflict checks.

## Reachability

Each `TM` can pick from or place into only modules connected to that TM by the equipment topology.

- Atmospheric-side TMs usually reach LP, AL, and conversion LLs.
- Vacuum-side TMs usually reach conversion LLs, transfer LLs, and PMs.

A pick/place targeting an unreachable module is illegal.

## Movement Time

Consecutive operations by the same robot must leave enough time for movement or rotation.

If the previous operation ends at `t1`, the next operation starts at `t2`, and required movement time is `d`, require:

```text
t2 - t1 >= d
```

The movement time may be constant, distance-based, angle-based, or a module-pair lookup.

## Pick / Place Duration

Each pick/place duration must match the equipment parameter:

```text
Pick.end - Pick.start = pick_time
Place.end - Place.start = place_time
```

`pick_time` and `place_time` may depend on TM, hand, module, side, or task.

## Robot Operation Mutual Exclusion

The same `TM` can execute only one pick/place operation at a time unless the equipment explicitly allows parallel operations.

For dual-arm TMs, two hands can hold wafers, but the robot usually still performs only one pick/place operation at a time.

## Hand Capacity

Each hand has capacity 1.

- `Pick.start` occupies the selected hand.
- The hand remains occupied by that wafer until the same wafer's `Place.end`.
- A hand cannot start a second pick while occupied.
- A place is legal only if the selected hand holds the placed wafer.

## Dual-Arm Geometry

Dual-arm TMs may have fixed geometry, often 180-degree symmetry.

At any time, if one hand points to a module, the other hand can point only to modules allowed by the geometry. Apply this constraint to simultaneous holding, exchange moves, and consecutive pick/place patterns that depend on hand orientation.

## Pick Preconditions

At `Pick.start`:

- Source chamber is occupied by the wafer.
- Wafer's required internal time at the source chamber is complete.
- Source chamber state permits pick.
- Robot can reach the source chamber.
- Selected hand is empty.

Source chamber capacity is released at `Pick.end`.

## Place Preconditions

At `Place.start`:

- Selected hand holds the wafer.
- Target chamber is allowed by the wafer route step.
- Target chamber has available capacity.
- Target chamber state permits place.
- Robot can reach the target chamber.

Target chamber capacity is occupied at `Place.start`; hand is released at `Place.end`.
