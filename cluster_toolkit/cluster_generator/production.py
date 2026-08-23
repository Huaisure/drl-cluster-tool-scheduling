from __future__ import annotations

import hashlib
import random
from pathlib import Path

from cluster_toolkit.validator import ValidatorSuite

from .corpus import InstanceCorpus, _json_bytes, _write_new_atomic
from .heuristic import build_safe_reference_schedule
from .pipeline import GeneratedInstance, InstanceGenerator, PERIODIC_RATIOS
from .pipeline_catalog import PipelineCatalog
from .pipeline_models import InstanceGenerationRequest, TopologyTemplate
from .problem_adapter import to_cluster_problem
from .production_models import (
    ProductionPlan,
    ProductionPlanEntry,
    ProductionRunSpec,
)
from .topology_family import (
    ATMOSPHERIC_FAMILY_ID,
    AtmosphericTopologyRequest,
    generate_atmospheric_topology,
)


RUN_SPEC_FILE = "run_spec.json"
RUN_PLAN_FILE = "plan.json"


def default_run_id(master_seed: int, instance_count: int) -> str:
    payload = f"atmospheric-archetype-v1:{master_seed}:{instance_count}".encode(
        "utf-8"
    )
    return f"run-{hashlib.sha256(payload).hexdigest()[:12]}"


def materialize_plan(run_root: str | Path, spec: ProductionRunSpec) -> ProductionPlan:
    """Create immutable topology snapshots and strictly validated instances."""

    source_catalog = PipelineCatalog.load(
        Path(__file__).parents[2] / "topologies",
        Path(__file__).parents[2] / "recipe_generation_profiles",
    )
    profile = source_catalog.profile(spec.profile_id)
    selected_topologies = (
        _selected_catalog_topologies(spec, source_catalog)
        if spec.schema_version == 2
        else ()
    )

    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / RUN_SPEC_FILE
    spec_bytes = _json_bytes(spec.model_dump(mode="json"))
    if spec_path.exists():
        if spec_path.read_bytes() != spec_bytes:
            raise FileExistsError(f"run_spec.json is immutable: {spec_path}")
    else:
        _write_new_atomic(spec_path, spec_bytes)

    topology_dir = root / "topologies"
    topology_dir.mkdir(exist_ok=True)
    profile_dir = root / "profiles"
    profile_dir.mkdir(exist_ok=True)
    _write_model_snapshot(
        profile_dir / f"{profile.profile_id}.json",
        profile.model_dump(mode="json"),
    )

    rng = random.Random(spec.master_seed)
    topologies: list[TopologyTemplate] = []
    topology_seeds: dict[str, int] = {}
    if spec.schema_version == 1:
        cell_counts = _weighted_quota_values(
            spec.topology_count,
            spec.cell_count_weights,
            rng,
        )
        for index, cell_count in enumerate(cell_counts):
            topology_seed = _derived_seed(spec.master_seed, "topology", index)
            topology = generate_atmospheric_topology(
                AtmosphericTopologyRequest(
                    seed=topology_seed,
                    cell_count=cell_count,
                )
            )
            _write_model_snapshot(
                topology_dir / f"{topology.topology_id}.json",
                topology.model_dump(mode="json"),
            )
            topologies.append(topology)
            topology_seeds[topology.topology_id] = topology_seed
    else:
        for index, topology in enumerate(selected_topologies):
            topology_seed = _derived_seed(spec.master_seed, "topology", index)
            _write_model_snapshot(
                topology_dir / f"{topology.topology_id}.json",
                topology.model_dump(mode="json"),
            )
            topologies.append(topology)
            topology_seeds[topology.topology_id] = topology_seed

    if spec.schema_version == 1:
        instance_topologies = [
            topologies[ordinal % len(topologies)]
            for ordinal in range(spec.instance_count)
        ]
    else:
        instance_topologies = _assign_instance_topologies(spec, topologies, rng)

    catalog = PipelineCatalog(
        topologies={topology.topology_id: topology for topology in topologies},
        profiles={profile.profile_id: profile},
    )
    generator = InstanceGenerator(
        catalog,
        wafer_ranges={
            scale: (interval.minimum, interval.maximum)
            for scale, interval in spec.wafer_ranges.items()
        },
    )
    corpus = InstanceCorpus(root)
    entries: list[ProductionPlanEntry] = []
    periodic_count = round(spec.instance_count * spec.periodic_fraction)
    periodic_flags = [True] * periodic_count + [False] * (
        spec.instance_count - periodic_count
    )
    rng.shuffle(periodic_flags)
    route_patterns = tuple(sorted(spec.route_pattern_weights))
    route_weights = tuple(
        spec.route_pattern_weights[pattern] for pattern in route_patterns
    )

    for ordinal in range(spec.instance_count):
        topology = instance_topologies[ordinal]
        recipe_count = spec.recipe_counts[ordinal % len(spec.recipe_counts)]
        wafer_scale = spec.wafer_scales[
            (ordinal // len(spec.recipe_counts)) % len(spec.wafer_scales)
        ]
        if len(topology.cell_order) == 1:
            route_pattern = "local"
        else:
            route_pattern = rng.choices(
                route_patterns,
                weights=route_weights,
                k=1,
            )[0]
        periodic_requested = periodic_flags[ordinal]
        periodic_ratio = (
            rng.choice(tuple(sorted(PERIODIC_RATIOS[recipe_count])))
            if periodic_requested
            else None
        )
        request = InstanceGenerationRequest(
            topology_id=topology.topology_id,
            profile_id=profile.profile_id,
            recipe_count=recipe_count,
            wafer_scale=wafer_scale,
            seed=_derived_seed(spec.master_seed, "instance", ordinal),
            periodic_ratio=periodic_ratio,
            route_pattern=route_pattern,
        )
        generated = generator.generate(request)
        problem = to_cluster_problem(generated.instance)
        witness = build_safe_reference_schedule(problem)
        report = ValidatorSuite(problem).validate(
            witness.actions,
            require_complete=True,
            exact_action_durations=True,
        )
        if not report.ok:
            details = "; ".join(issue.message for issue in report.issues[:8])
            raise RuntimeError(
                f"generated instance {generated.instance_id} failed serial witness: "
                f"{details}"
            )
        generated = GeneratedInstance(
            instance=generated.instance,
            metadata={
                **generated.metadata,
                "generation_validation": "VALID",
                "serial_witness_saved": False,
                "topology_cell_count": len(topology.cell_order),
                "run_id": spec.run_id,
                "master_seed": spec.master_seed,
                "generation_ordinal": ordinal,
                "topology_seed": topology_seeds[topology.topology_id],
                "topology_archetype_id": topology.archetype_id,
                "robot_arm_profile_id": topology.arm_profile_id,
                "instance_seed": request.seed,
            },
        )
        corpus.materialize(generated)
        entries.append(
            ProductionPlanEntry(
                ordinal=ordinal,
                request=request,
                instance_id=generated.instance_id,
                topology_cell_count=len(topology.cell_order),
                topology_seed=topology_seeds[topology.topology_id],
                topology_archetype_id=topology.archetype_id,
                robot_arm_profile_id=topology.arm_profile_id,
                periodic_requested=periodic_requested,
                route_pattern=route_pattern,
            )
        )

    plan = ProductionPlan(run_id=spec.run_id, entries=tuple(entries))
    plan_path = root / RUN_PLAN_FILE
    plan_bytes = _json_bytes(plan.model_dump(mode="json"))
    if plan_path.exists():
        if plan_path.read_bytes() != plan_bytes:
            raise FileExistsError(f"plan.json is immutable: {plan_path}")
    else:
        _write_new_atomic(plan_path, plan_bytes)
    return plan


def load_run(run_root: str | Path) -> tuple[ProductionRunSpec, ProductionPlan]:
    root = Path(run_root)
    spec = ProductionRunSpec.model_validate_json(
        (root / RUN_SPEC_FILE).read_text(encoding="utf-8")
    )
    plan = ProductionPlan.model_validate_json(
        (root / RUN_PLAN_FILE).read_text(encoding="utf-8")
    )
    if plan.run_id != spec.run_id:
        raise ValueError("plan.json run_id does not match run_spec.json")
    return spec, plan


def _derived_seed(master_seed: int, stage: str, ordinal: int) -> int:
    digest = hashlib.sha256(
        f"{master_seed}:{stage}:{ordinal}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _selected_catalog_topologies(
    spec: ProductionRunSpec,
    catalog: PipelineCatalog,
) -> tuple[TopologyTemplate, ...]:
    candidates = tuple(
        topology
        for topology in catalog.topologies
        if topology.family_id == ATMOSPHERIC_FAMILY_ID
        and topology.archetype_id is not None
        and topology.arm_profile_id is not None
    )
    if not candidates:
        raise ValueError("topology catalog has no atmospheric archetype JSON files")

    available_archetypes = {topology.archetype_id for topology in candidates}
    requested_archetypes = (
        spec.topology_archetypes
        if spec.topology_archetypes
        else tuple(sorted(available_archetypes))
    )
    unknown = set(requested_archetypes) - available_archetypes
    if unknown:
        raise ValueError(f"unknown topology archetypes in catalog: {sorted(unknown)}")

    by_archetype: dict[str, list[TopologyTemplate]] = {}
    seen_variants: set[tuple[str, str]] = set()
    for topology in candidates:
        assert topology.archetype_id is not None
        assert topology.arm_profile_id is not None
        if topology.archetype_id not in requested_archetypes:
            continue
        cell_count = len(topology.cell_order)
        if spec.cell_count_weights[cell_count] <= 0:
            continue
        variant = (topology.archetype_id, topology.arm_profile_id)
        if variant in seen_variants:
            raise ValueError(f"duplicate topology catalog variant: {variant}")
        seen_variants.add(variant)
        by_archetype.setdefault(topology.archetype_id, []).append(topology)

    missing_cells = [
        cell_count
        for cell_count, weight in spec.cell_count_weights.items()
        if weight > 0
        and not any(
            len(topology.cell_order) == cell_count
            for variants in by_archetype.values()
            for topology in variants
        )
    ]
    if missing_cells:
        raise ValueError(
            "positive cell_count_weights require catalog topology JSON for Cells "
            f"{missing_cells}"
        )
    if spec.topology_count > sum(len(items) for items in by_archetype.values()):
        raise ValueError(
            "topology_count exceeds the unique catalog variants enabled by this run"
        )

    archetype_order = tuple(
        archetype_id
        for archetype_id in requested_archetypes
        if archetype_id in by_archetype
    )
    for variants in by_archetype.values():
        cell_counts = {len(topology.cell_order) for topology in variants}
        if len(cell_counts) != 1:
            raise ValueError("one archetype cannot span multiple Cell counts")
        variants.sort(key=lambda topology: topology.arm_profile_id or "")
    base_topologies = tuple(by_archetype[item][0] for item in archetype_order)
    mandatory: list[TopologyTemplate] = []
    for cell_count in (1, 2, 3):
        if spec.cell_count_weights[cell_count] <= 0:
            continue
        mandatory.append(
            next(
                topology
                for topology in base_topologies
                if len(topology.cell_order) == cell_count
            )
        )
    ordered = [
        *mandatory,
        *(topology for topology in base_topologies if topology not in mandatory),
        *(
            topology
            for archetype_id in archetype_order
            for topology in by_archetype[archetype_id][1:]
        ),
    ]
    return tuple(ordered[: spec.topology_count])


def _assign_instance_topologies(
    spec: ProductionRunSpec,
    topologies: list[TopologyTemplate],
    rng: random.Random,
) -> list[TopologyTemplate]:
    by_cell_and_archetype: dict[int, dict[str, list[TopologyTemplate]]] = {
        cell_count: {} for cell_count in (1, 2, 3)
    }
    for topology in topologies:
        cell_count = len(topology.cell_order)
        assert topology.archetype_id is not None
        by_cell_and_archetype[cell_count].setdefault(
            topology.archetype_id,
            [],
        ).append(topology)
    for archetypes in by_cell_and_archetype.values():
        for variants in archetypes.values():
            variants.sort(key=lambda topology: topology.arm_profile_id or "")

    weights = {
        cell_count: weight
        for cell_count, weight in spec.cell_count_weights.items()
        if by_cell_and_archetype[cell_count]
    }
    cell_counts = _weighted_quota_values(spec.instance_count, weights, rng)
    positions_by_cell = {
        cell_count: [
            ordinal
            for ordinal, assigned_cell_count in enumerate(cell_counts)
            if assigned_cell_count == cell_count
        ]
        for cell_count in (1, 2, 3)
    }
    result: list[TopologyTemplate | None] = [None] * spec.instance_count
    for cell_count, positions in positions_by_cell.items():
        if not positions:
            continue
        archetypes = by_cell_and_archetype[cell_count]
        archetype_ids = _uniform_quota_values(
            len(positions),
            tuple(sorted(archetypes)),
            rng,
        )
        offsets = {archetype_id: 0 for archetype_id in archetypes}
        for ordinal, archetype_id in zip(positions, archetype_ids, strict=True):
            choices = archetypes[archetype_id]
            offset = offsets[archetype_id]
            result[ordinal] = choices[offset % len(choices)]
            offsets[archetype_id] += 1
    assert all(topology is not None for topology in result)
    return [topology for topology in result if topology is not None]


def _uniform_quota_values(
    count: int,
    values: tuple[str, ...],
    rng: random.Random,
) -> list[str]:
    quotient, remainder = divmod(count, len(values))
    result = [value for value in values for _ in range(quotient)]
    result.extend(values[:remainder])
    rng.shuffle(result)
    return result


def _weighted_quota_values(
    count: int,
    weights: dict[int, float],
    rng: random.Random,
) -> list[int]:
    total = sum(weights.values())
    exact = {key: count * value / total for key, value in weights.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(
        weights,
        key=lambda item: (-(exact[item] - quotas[item]), item),
    )[:remaining]:
        quotas[key] += 1
    values = [key for key in sorted(quotas) for _ in range(quotas[key])]
    rng.shuffle(values)
    return values


def _write_model_snapshot(path: Path, value: object) -> None:
    payload = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable snapshot differs: {path}")
        return
    _write_new_atomic(path, payload)
