# Subject-oriented Validator Scaffold

This package validates one action sequence from three independent views.
The sibling `problem/` package reads and validates machine/route JSON; it is
kept outside this package so the two components can be maintained separately.
`ValidatorSuite` groups the input actions and creates one validator instance for
every concrete subject:

```text
actions
  -> PM1 actions -> ModuleValidator("PM1")
  -> PM2 actions -> ModuleValidator("PM2")
  -> TM1 actions -> RobotValidator("TM1")
  -> wafer A-0 actions -> WaferValidator(("A", 0))
  -> merge reports
```

Typical usage:

```python
from problem import load_problem
from validator import ValidatorSuite

problem = load_problem("validator/examples/naura_task1.json")
report = ValidatorSuite(problem).validate(actions)
```

Each validator owns exactly one subject ID, its relevant problem configuration,
its time-ordered actions, and its subject-local initial slice. `ModuleValidator` checks
module capacity at Place.start and Pick.end. Capacity defaults to 25 for LP and
1 for every other module; an explicit positive `capacity` emits a warning and
overrides that default.
`RobotValidator` checks action mutual exclusion and single/dual-arm total
capacity, movement time between actions at different modules, and minimum
place/pick durations.
`WaferValidator` checks route order and overlap among Pick, Place, and
PM-processing intervals.

Wafer initial state is defined only by `problem.initial_state.wafers`. Module
occupants and robot-arm occupants are derived from each wafer's location; they
are not accepted as separate inputs. `InitialState.to_snapshot()` creates one
read-only `InitialSnapshot` per validation run, and `ValidatorSuite` distributes
its subject-local slices. In problem JSON, an initial `wafer_index` is a string
expression such as `"3"`, `"1-5"`, or `"1,3-5,8"`; parsing expands it into
individual integer-indexed wafers before snapshot creation.
An optional `problem.initial_state.robots[robot_id].position_module_id` sets a
robot's physical start position. Missing or `null` means `anywhere`; movement
before only the first robot action is therefore unconstrained.

Input actions are normalized exactly once before grouping. Internally the
validator uses `pick` and `place`; external `unload` and `load` remain accepted
aliases. There is no generic constraint layer or shared replay base class. Module,
Robot, and Wafer rules will live directly in their corresponding validator.
Shared mechanical helpers, such as interval overlap and inclusive time-window
checks, live under `common/`. Cleaning and Pump/Vent remain out of scope.

Chinese design documentation is maintained in
[`docs/README-zh.md`](docs/README-zh.md).

Use Python 3.10 or newer to run both packages' framework tests:

```bash
pytest -q validator/tests problem/tests
```
