# Cluster Tool Scheduling

This context defines the shared language for compiling heterogeneous cluster-tool constraints into one auditable scheduling semantics. The vocabulary deliberately separates domain meaning from model inputs and implementation choices.

## Semantic foundation

**Unified Discrete-Event State Transition System**:
The canonical scheduling model in which the system state is the product of orthogonal equipment, wafer, resource, and obligation states, and changes only at explicit event boundaries.
_Avoid_: Unified finite-state machine, giant state enum, monolithic state machine

**Stable State**:
A state after all consequences at the current time tick have reached a fixed point and before a new scheduling decision is requested.
_Avoid_: Snapshot, observation

**Event Boundary**:
An instantaneous start, end, trigger, or deadline boundary at which the scheduling state may change.
_Avoid_: Action step, frame

**Decision Epoch**:
A stable state in which at least one intent is enabled or the schedule is complete or infeasible.
_Avoid_: Every tick, model timestep

**Time Tick**:
The exact integer unit used by the semantic kernel after external durations have been converted from their declared physical unit.
_Avoid_: Float time, simulation frame

## Scheduling language

**Operator Template**:
A declarative description of a reusable operation, including its boundaries, guards, resource claims, effects, triggers, and obligations.
_Avoid_: Model action, hard-coded constraint branch

**Intent**:
A parameterized scheduling choice submitted at a decision epoch; one intent may expand into several operator boundaries and events.
_Avoid_: Raw event, primitive action

**Event**:
A concrete, timestamped boundary produced by expanding an accepted intent or by an automatic kernel transition.
_Avoid_: Intent, command

**Schedule**:
The auditable collection of concrete events and intervals, including automatic Process events and explicit Clean, Pump, and Vent events.
_Avoid_: Action history, policy trace

**Guard**:
A condition that must hold at an operator boundary for the boundary to occur legally.
_Avoid_: Constraint penalty, action mask rule

**Invariant**:
A condition that must hold throughout every reachable scheduling state or claimed interval.
_Avoid_: Guard, objective

**Trigger**:
A state change that creates an automatic event or obligation when its declared condition becomes true.
_Avoid_: Reward signal, policy hint

**Obligation**:
A future requirement created by an event, with a satisfaction condition and optionally a deadline.
_Avoid_: Soft penalty, hidden constraint

## Resource semantics

**Resource Claim**:
A request by an operator for exclusive or capacity-limited use of a resource over an interval.
_Avoid_: Lock, occupancy flag

**Resource Lease**:
An interval of resource ownership whose lifetime may span several events, such as a wafer occupying a slot or a robot hand holding a wafer.
_Avoid_: Reservation, transient claim

**Reservation**:
A commitment of future resource capacity that prevents conflicting intents before the resource lease begins.
_Avoid_: Lease, prediction

**Enabled Intent**:
An intent whose immediate boundary guards hold in the current stable state.
_Avoid_: Valid action, feasible schedule

**Committable Intent**:
An enabled intent whose complete declared boundary bundle can be reserved without a known conflict or violated obligation.
_Avoid_: Enabled intent, globally feasible intent

**Completable State**:
A stable state from which at least one legal continuation reaches all required terminal conditions.
_Avoid_: Non-deadlocked state, enabled state

## Load-lock semantics

**Pressure State**:
The load lock's atmosphere/vacuum condition and any Pump or Vent transition currently changing it; it is independent of wafer thermal readiness.
_Avoid_: Load-lock state, wafer state

**Wafer Thermal State**:
A wafer-specific cooling or thermal-readiness condition while the wafer occupies a load-lock slot; it may progress concurrently with a pressure transition.
_Avoid_: Pressure state, combined load-lock phase

**Interface Side**:
The atmosphere-side or vacuum-side accessibility required by a transfer boundary after the relevant pressure transition is complete.
_Avoid_: Robot side state, load-lock mode
