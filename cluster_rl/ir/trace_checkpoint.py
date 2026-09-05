"""Write an audited, human-readable decision trace for one SFT case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .env import IRSchedulingEnv
from .graph import FEATURE_VERSION
from .network import IRActorCritic, collate_graphs
from .sft import SFTConfig, _runtime_config
from .sft_data import _candidate_key, load_sft_cases


def trace_case(checkpoint: Path, manifest: Path, instance_id: str, output: Path) -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("checkpoint feature protocol mismatch")
    config = SFTConfig(**saved["sft_config"])
    model = IRActorCritic(config.width, config.layers).to(config.device)
    model.load_state_dict(saved["model"])
    model.eval()
    cases = load_sft_cases([manifest])
    case = next(
        (item for item in cases if item.metadata["instance_id"] == instance_id),
        None,
    )
    if case is None:
        raise ValueError(f"instance not found: {instance_id}")
    env = IRSchedulingEnv(case.problem, **_runtime_config(config).env_options())
    observation, _ = env.reset(seed=config.seed)
    rows = []
    with torch.no_grad():
        while env.reason is None:
            output_value = model(collate_graphs([observation], config.device))
            order = output_value.logits[0].argsort(descending=True).tolist()
            def describe(index: int) -> dict[str, object]:
                if index == len(env.frame.intents):
                    return {
                        "action": "wait",
                        "until_tick": env.wait_tick,
                        "logit": float(output_value.logits[0, index]),
                    }
                candidate = env.frame.intents[index]
                return {
                    "action": "intent",
                    "candidate_key": candidate.candidate_key,
                    "semantic_key": list(_candidate_key(candidate)),
                    "duration": candidate.duration_ticks,
                    "logit": float(output_value.logits[0, index]),
                }
            rows.append({
                "decision": env.decisions,
                "tick": env.snapshot.tick,
                "selected": describe(order[0]),
                "ranked_candidates": [describe(index) for index in order[:5]],
            })
            observation, _, _, _, info = env.step(int(order[0]))
    audit = env.audit()
    if not audit.ok:
        raise RuntimeError(f"trace audit failed: {audit.issues}")
    result = {
        "instance_id": instance_id,
        "problem_hash": case.problem.problem_hash,
        "termination_reason": info["termination_reason"],
        "success": bool(info["success"]),
        "elapsed_seconds": info["elapsed_seconds"],
        "decisions": rows,
        "expert_actions": list(case.actions),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    temporary.replace(output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = trace_case(
        args.checkpoint, args.manifest, args.instance_id, args.output,
    )
    print(json.dumps({
        "instance_id": result["instance_id"],
        "success": result["success"],
        "termination_reason": result["termination_reason"],
        "decision_count": len(result["decisions"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
