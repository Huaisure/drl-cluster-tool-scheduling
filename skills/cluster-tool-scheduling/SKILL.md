---
name: cluster-tool-scheduling
description: Domain guide for cluster tool semiconductor scheduling. Use when Codex needs to understand, analyze, validate, or implement code for cluster tool scheduling tasks, including Pick/Place/Clean/Pump/Vent timing, TM robot constraints, chamber occupancy, PM cleaning rules, CP-SAT scheduling models, action sequence validators, result validators, or deadlock/time/resource conflict analysis.
---

# Cluster Tool Scheduling

Use this skill to reason about cluster tool scheduling as timed events that change resource occupancy and equipment state.

## Core Rule

Treat a schedule as a sequence of timed events. Each event has a start time, end time, preconditions, resource effects, and state effects. A schedule is valid only if resource capacities, wafer identities, chamber states, robot states, and timing constraints remain consistent at every event boundary.

Use `[start, end)` semantics unless the user's project states otherwise.

## Reference Routing

Load only the reference needed for the user's task:

- To understand or explain the whole task, read `references/problem-framework.md`.
- To implement or review action legality, result validation, or Pick/Place/Clean/Pump/Vent timing, read `references/action-semantics.md` first.
- To handle TM reachability, movement, hand occupancy, dual-arm geometry, or robot timing, read `references/robot-constraints.md`.
- To handle LP/AL/PM/LL occupancy, chamber capacity, valve/interface use, or residency, read `references/chamber-constraints.md`.
- To handle PM cleaning triggers, cleaning timing, cleaning priority, or cleaning-state replay, read `references/cleaning-constraints.md`.

For broad scheduler or validator work, read `references/action-semantics.md` plus the subsystem references touched by the requested change. Read `references/problem-framework.md` only when the user asks for the whole problem or when subsystem boundaries are unclear.

## Implementation Guidance

- Keep modeling and validation conceptually separate: validators should check the problem semantics, not mirror a solver's internal variables.
- State event boundaries explicitly in code and documentation, especially `Pick.start`, `Pick.end`, `Place.start`, `Place.end`, `Clean.start`, `Clean.end`, `Pump/Vent.start`, and `Pump/Vent.end`.
- When a schedule uses candidate modules, reason at the level provided by the schedule; do not invent physical-module assignments unless the data provides them or the user asks for a materialization step.
