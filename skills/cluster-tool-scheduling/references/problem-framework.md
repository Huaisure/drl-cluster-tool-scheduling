# Cluster Tool Scheduling Problem Framework

Use this full framework when the user asks to understand the whole task or when a broad scheduler/validator change crosses robot, chamber, and cleaning boundaries.

## Core Idea

A schedule is a set of timed events. Each event occupies or releases resources and may update equipment state. A schedule is legal only if every resource's occupancy count, occupancy identity, and state remain valid at every event boundary.

Use `[start, end)` interval semantics unless the project specifies otherwise.

## Equipment Regions

Cluster tools usually have:

- Atmosphere region: LP, AL, atmosphere-side TM, and conversion LLs.
- Vacuum region: vacuum-side TM, PMs, and vacuum transfer LLs.

Wafers move between atmosphere and vacuum only through conversion LLs.

## Main Objects

- LP: cassette interface and wafer start/end slots.
- AL: alignment chamber.
- TM: robot moving wafers between reachable modules.
- Conversion LL: atmosphere/vacuum bridge with state transitions.
- Vacuum transfer LL: vacuum-internal transfer chamber, usually zero process time and no cleaning.
- PM: process chamber, possibly requiring cleaning.
- Hand, slot, valve, interface: lower-level resources when represented.

## Route Semantics

Each wafer follows an ordered route. A route node may be a single module or a set of candidate modules. The wafer chooses one allowed module for that node and cannot skip route nodes or leave before the node's required time is complete.

## Event Semantics Summary

- `Place.start`: occupies target chamber capacity.
- `Place.end`: wafer is in target chamber; robot hand is released.
- `Pick.start`: occupies robot hand and source interface; requires source internal time complete.
- `Pick.end`: releases source chamber capacity.
- `Clean.start`: PM cleaning begins; PM must be empty.
- `Clean.end`: PM cleaning finishes.
- `Pump/Vent.start`: LL state transition begins; LL must be in source state.
- `Pump/Vent.end`: LL state transition completes; LL state becomes target state.

## Robot Rules

- TM can access only topology-reachable modules.
- Consecutive operations by the same TM must respect movement time.
- Pick/place durations must match equipment parameters.
- The same TM usually performs only one pick/place action at a time.
- Each hand has capacity 1 from `Pick.start` until the wafer's `Place.end`.
- Dual-arm geometry must be respected when applicable.

## Chamber Rules

- A wafer occupies a chamber from entering `Place.start` to leaving `Pick.end`.
- Chamber occupancy count must not exceed capacity.
- Required internal time must fit between `Place.end` and `Pick.start`.
- Conversion LL pick/place requires LL state compatible with the robot side.
- Vacuum transfer LL has no Pump/Vent requirement and does not trigger PM cleaning.
- Valve/interface conflicts must be checked if represented.

## Cleaning Rules

Cleaning is PM-exclusive and starts only when the PM is empty.

Common triggers:

- Type A: idle-time clean.
- Type B: process-switch clean.
- Type C: wafer-count clean.

Triggered cleaning must occur before the next incompatible wafer enters the PM. Priority must follow the problem definition.

## Deadlock

A state is deadlocked if no legal next event can proceed, or if progress requires violating time, capacity, identity, or state rules.
