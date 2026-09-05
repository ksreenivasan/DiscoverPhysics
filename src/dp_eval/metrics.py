from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRICES_PER_MILLION = {
    "gpt-5.6-sol": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "gemini-3.7-flash": (0.75, 3.75),
    "qwen/qwen3.5-397b-a17b": (0.50, 3.60),
    "z-ai/glm-5.1": (0.966, 3.036),
    "nvidia/nemotron-3-ultra-550b-a55b": (0.60, 3.60),
}


def aggregate(run_dir: Path, label: str = "Public-Pilot-4") -> dict[str, Any]:
    trials = []
    for path in sorted((run_dir / "raw").glob("*/*/seed-*/trial.json")):
        row = json.loads(path.read_text())
        row.setdefault(
            "item_key",
            f"{_safe(row['lane'])}/{row['world']}/seed-{int(row['seed'])}",
        )
        trials.append(row)
    manifest_path = run_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"trial_count": len(trials), "items": []}
    )
    expected_items = manifest.get("items") or []
    if not trials and not expected_items:
        raise ValueError(f"no trial.json files under {run_dir / 'raw'}")
    observed = {row["item_key"]: row for row in trials}
    if len(observed) != len(trials):
        raise ValueError("duplicate deterministic item keys in trial artifacts")
    missing_items = [item for item in expected_items if item["item_key"] not in observed]
    for item in missing_items:
        trials.append(
            {
                **item,
                "status": "unresolved",
                "error": {
                    "type": "MissingArtifact",
                    "message": "manifest item has no trial artifact and remains retryable",
                },
            }
        )

    by_lane: dict[str, list[dict]] = defaultdict(list)
    for row in trials:
        by_lane[row["lane"]].append(row)
    lane_results = {}
    for lane, rows in sorted(by_lane.items()):
        lane_results[lane] = _lane_metrics(rows, run_dir)

    result = {
        "schema_version": 1,
        "label": label,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "worlds": sorted({row["world"] for row in trials}),
        "seeds": sorted({row["seed"] for row in trials}),
        "trial_count": len(trials),
        "manifest_trial_count": int(manifest.get("trial_count", len(trials))),
        "persisted_trial_count": len(observed),
        "unresolved_missing_items": [item["item_key"] for item in missing_items],
        "run_complete": not missing_items,
        "all_items_scored": not missing_items and all(
            row.get("status") == "completed" for row in trials
        ),
        "lanes": lane_results,
    }
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    _write_csv(aggregate_dir / "trials.csv", trials)
    (aggregate_dir / "summary.md").write_text(render_markdown(result))
    return result


def _lane_metrics(rows: list[dict], run_dir: Path) -> dict[str, Any]:
    worlds = sorted({row["world"] for row in rows})
    by_world = {world: sorted((r for r in rows if r["world"] == world), key=lambda r: r["seed"]) for world in worlds}
    expected = {}
    for k in (1, 3, 5):
        contributions = []
        for world_rows in by_world.values():
            successes = sum(bool(r.get("joint_pass")) for r in world_rows)
            n = len(world_rows)
            if n < k:
                contributions.append(0.0)
            else:
                contributions.append(1.0 - math.comb(n - successes, k) / math.comb(n, k) if n - successes >= k else 1.0)
        expected[f"pass_at_{k}"] = 100.0 * sum(contributions) / len(contributions)

    finite_mse = [float(r["normalized_mse"]) for r in rows if isinstance(r.get("normalized_mse"), (int, float)) and math.isfinite(r["normalized_mse"]) and r["normalized_mse"] > 0]
    scores = [float(r.get("explanation_score") or 0.0) for r in rows]
    usage = _usage_for_lane(rows[0]["lane"], run_dir)
    model_id = rows[0]["model_id"]
    price = PRICES_PER_MILLION.get(model_id)
    estimated_cost = None
    if price and usage["input_tokens"] is not None and usage["output_tokens"] is not None:
        estimated_cost = usage["input_tokens"] / 1e6 * price[0] + usage["output_tokens"] / 1e6 * price[1]

    return {
        "model_id": model_id,
        "provider": rows[0]["provider"],
        "provider_backend": rows[0].get("provider_backend"),
        "requested_reasoning": "high",
        "completed_trials": sum(r["status"] == "completed" for r in rows),
        "failed_trials": sum(r["status"] != "completed" for r in rows),
        "joint_passes": sum(bool(r.get("joint_pass")) for r in rows),
        "trial_count": len(rows),
        "trial_joint_pass_rate": 100.0 * sum(bool(r.get("joint_pass")) for r in rows) / len(rows),
        **expected,
        "geometric_mean_normalized_mse": math.exp(sum(math.log(v) for v in finite_mse) / len(finite_mse)) if finite_mse else None,
        "mean_explanation_score": sum(scores) / len(scores),
        "per_world": {
            world: {
                "joint_successes": sum(bool(r.get("joint_pass")) for r in world_rows),
                "attempts": len(world_rows),
                "normalized_mse": [r.get("normalized_mse") for r in world_rows],
                "explanation_scores": [r.get("explanation_score") for r in world_rows],
            }
            for world, world_rows in by_world.items()
        },
        "usage": usage,
        "estimated_solver_cost_usd": estimated_cost,
    }


def _usage_for_lane(lane: str, run_dir: Path) -> dict[str, Any]:
    safe_lane = "".join(c if c.isalnum() or c in "_.-" else "_" for c in lane)
    records = []
    for path in (run_dir / "raw" / safe_lane).glob("*/seed-*/usage.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    solver_records = [r for r in records if r.get("call_role", "solver") == "solver"]
    judge_records = [r for r in records if r.get("call_role") == "judge"]
    def total(field: str):
        vals = [r.get(field) for r in solver_records if isinstance(r.get(field), int)]
        return sum(vals) if vals else None
    return {
        "calls": len(solver_records),
        "judge_calls": len(judge_records),
        "input_tokens": total("input_tokens"),
        "cached_input_tokens": total("cached_input_tokens"),
        "output_tokens": total("output_tokens"),
        "reasoning_tokens": total("reasoning_tokens"),
        "latency_ms": total("latency_ms"),
        "effective_reasoning_values": sorted({r.get("effective_reasoning") for r in solver_records if r.get("effective_reasoning")}),
        "all_solver_model_ids_verified": all(r.get("model_identity_verified", False) for r in solver_records),
        "all_solver_reasoning_settings_verified": all(r.get("reasoning_setting_verified", False) for r in solver_records),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['label']} report",
        "",
        f"Run: `{result['run_id']}`  ",
        f"Generated: {result['generated_utc']}  ",
        f"Worlds: {', '.join(result['worlds'])}  ",
        f"Seeds: {', '.join(map(str, result['seeds']))}  ",
        f"Manifest accounted: **{result['run_complete']}**; all items scored: **{result['all_items_scored']}**",
        "",
        "> Internal stratified public pilot; not an official 22-world DiscoverPhysics score.",
        "",
        "| Lane | Completed | Joint passes | Pass@1 | Pass@3 | Pass@5 | Geom. norm MSE | Mean explanation | Est. solver spend |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lane, row in result["lanes"].items():
        cost = "n/a" if row["estimated_solver_cost_usd"] is None else f"${row['estimated_solver_cost_usd']:.2f}"
        gm = "n/a" if row["geometric_mean_normalized_mse"] is None else f"{row['geometric_mean_normalized_mse']:.4g}"
        lines.append(
            f"| {lane} | {row['completed_trials']}/{row['trial_count']} | {row['joint_passes']}/{row['trial_count']} | "
            f"{row['pass_at_1']:.1f}% | {row['pass_at_3']:.1f}% | {row['pass_at_5']:.1f}% | {gm} | "
            f"{row['mean_explanation_score']:.3f} | {cost} |"
        )
    lines += ["", "## Per-world results", ""]
    for lane, row in result["lanes"].items():
        lines.append(f"### {lane} (`{row['model_id']}`, reasoning: high)")
        lines.append("")
        lines.append("| World | Joint successes | Normalized MSE by seed | Explanation by seed |")
        lines.append("|---|---:|---|---|")
        for world, values in row["per_world"].items():
            lines.append(
                f"| {world} | {values['joint_successes']}/{values['attempts']} | "
                f"{_fmt_list(values['normalized_mse'])} | {_fmt_list(values['explanation_scores'])} |"
            )
        lines.append("")
    lines += [
        "## Interpretation limits",
        "",
        "- Pass@k is computed exactly without replacement from the five attempts for these four public worlds.",
        "- Missing/failed attempts remain failures in the denominator.",
        "- Scores must not be extrapolated to the 22-world benchmark or compared directly with full-suite leaderboard rows.",
        "- Spend is estimated from recorded token usage and the execution-day catalog snapshot; provider billing is authoritative.",
    ]
    return "\n".join(lines) + "\n"


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)


def _fmt_list(values: list[Any]) -> str:
    return ", ".join("n/a" if v is None else f"{float(v):.3g}" for v in values)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "lane", "model_id", "provider", "provider_backend", "world", "seed", "status",
        "normalized_mse", "explanation_score", "trajectory_pass", "explanation_pass", "joint_pass",
        "rounds_used", "experiment_count", "fit_request_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
