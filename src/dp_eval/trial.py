from __future__ import annotations

import hashlib
import json
import math
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dp_eval.adapters import Adapter
from dp_eval.sandbox import SandboxClient

WORLD_VARS = {
    "gravity": 4.283,
    "oscillator": 6.332,
    "extra_dimensions": 4.248,
    "dark_matter": 63.303,
    "three_species": 28.717,
}


def run_trial(spec: dict[str, Any]) -> dict[str, Any]:
    run_id = spec["run_id"]
    lane = spec["lane"]
    world_name = spec["world"]
    seed = int(spec["seed"])
    artifact_root = Path(os.environ.get("DP_ARTIFACT_ROOT", "/artifacts"))
    trial_dir = artifact_root / run_id / "raw" / _safe(lane) / world_name / f"seed-{seed}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    usage_path = trial_dir / "usage.jsonl"
    adapter = Adapter(usage_path)
    started = _utc_now()

    try:
        result = _run_trial_inner(spec, trial_dir, adapter)
        status = "completed"
        error = None
    except Exception as exc:
        result = {}
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=12)}
        (trial_dir / "error.json").write_text(json.dumps(error, indent=2))

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": spec["phase"],
        "lane": lane,
        "model_id": spec["model_id"],
        "provider": spec["provider"],
        "provider_backend": spec.get("provider_backend") or ("auto" if spec["provider"] == "openrouter" else "native"),
        "requested_reasoning": "high",
        "world": world_name,
        "world_visibility": "public",
        "seed": seed,
        "status": status,
        "started_utc": started,
        "finished_utc": _utc_now(),
        "max_rounds": spec["max_rounds"],
        "max_tokens": spec["max_tokens"],
        "engine": "nbody",
        "noise_frac": spec["noise_frac"],
        "noise_std": spec["noise_frac"] * math.sqrt(WORLD_VARS[world_name]),
        "critic": False,
        "random_experiments": False,
        "mid_round_mse_fit": True,
        "judge_model": "claude-opus-4-6",
        "trajectory_threshold": 0.1,
        "explanation_threshold": 0.9,
        "error": error,
        **result,
    }
    (trial_dir / "trial.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def _run_trial_inner(spec: dict[str, Any], trial_dir: Path, adapter: Adapter) -> dict[str, Any]:
    from scienceagent import llm_client
    from scienceagent.agent import DiscoveryAgent
    from scienceagent.evaluator import ExplanationJudge, _extract_training_trajectories
    from scienceagent.trajectory_logger import TrajectoryLogger, make_run_id
    from scienceagent.worlds import get_world
    import scienceagent.mse_fitting as mse_fitting

    llm_client.complete = adapter.complete
    noise_std = spec["noise_frac"] * math.sqrt(WORLD_VARS[spec["world"]])
    world = get_world(spec["world"], engine="nbody", noise_std=noise_std, noise_seed=spec["seed"])
    executor = world["executor"]
    csv_path = trial_dir / "trajectory.csv"
    logger = TrajectoryLogger(
        world=spec["world"], executor=executor, csv_path=csv_path,
        run_id=make_run_id(spec["model_id"]),
    )
    sandbox = SandboxClient(timeout_s=900)
    original_fit_law = mse_fitting.fit_law

    def sandbox_fit_law(*, law_source, world, csv_path, run_id):
        return sandbox.request({
            "action": "fit_law", "law_source": law_source, "world": world,
            "csv_path": str(csv_path), "run_id": run_id,
        })

    mse_fitting.fit_law = sandbox_fit_law
    try:
        agent = DiscoveryAgent(
            model=spec["model_id"], executor=executor, mission=world["mission"],
            max_tokens=spec["max_tokens"], verbose=False, show_experiment_output=False,
            system_prompt_path=world["system_prompt"], instructions_path=world["instructions"],
            law_stub=world["law_stub"], experiment_format=world["experiment_format"],
            critic=None, max_rounds=spec["max_rounds"], min_rounds=2,
            random_experiments=False, trajectory_logger=logger, no_mse=False,
        )
        law_source = agent.run()
    finally:
        mse_fitting.fit_law = original_fit_law

    (trial_dir / "transcript.json").write_text(json.dumps(agent.conversation_log, indent=2, allow_nan=True))
    if law_source is None:
        return {
            "rounds_used": len(agent.conversation_log), "experiment_count": _experiment_count(agent.conversation_log),
            "final_law_sha256": None, "explanation_sha256": _hash_text(agent.discovered_explanation),
            "raw_mse": None, "normalized_mse": None, "explanation_score": 0.0,
            "trajectory_pass": False, "explanation_pass": False, "joint_pass": False,
            "evaluation_status": "no_final_law",
        }

    (trial_dir / "final_law.py").write_text(law_source)
    training = _extract_training_trajectories(agent.conversation_log)
    evaluation = sandbox.request({
        "action": "evaluate", "world": spec["world"], "engine": "nbody",
        "noise_std": noise_std, "noise_seed": spec["seed"],
        "law_source": law_source, "training_trajectories": training,
    })
    (trial_dir / "evaluation_trajectory.json").write_text(json.dumps(evaluation, indent=2, allow_nan=True))

    judge = ExplanationJudge(judge_model="claude-opus-4-6", max_tokens=1024)
    explanation = judge.score(
        agent_explanation=agent.discovered_explanation,
        optimal_explanation=world.get("optimal_explanation", ""),
        rubric=world.get("explanation_rubric", ""), verbose=False,
    )
    (trial_dir / "evaluation_explanation.json").write_text(json.dumps(explanation, indent=2, allow_nan=True))

    raw_mse = evaluation.get("mean_pos_error")
    normalized = raw_mse / WORLD_VARS[spec["world"]] if isinstance(raw_mse, (int, float)) and math.isfinite(raw_mse) else None
    score = explanation.get("score")
    trajectory_pass = normalized is not None and normalized < 0.1
    explanation_pass = isinstance(score, (int, float)) and score >= 0.9
    return {
        "rounds_used": len(agent.conversation_log),
        "experiment_count": _experiment_count(agent.conversation_log),
        "fit_request_count": sum(1 for row in agent.conversation_log if row.get("mse_fit_input") is not None),
        "final_law_sha256": _hash_text(law_source),
        "explanation_sha256": _hash_text(agent.discovered_explanation),
        "raw_mse": raw_mse,
        "normalized_mse": normalized,
        "explanation_score": score,
        "explanation_raw_score": explanation.get("raw_score"),
        "judge_error": explanation.get("error"),
        "trajectory_pass": trajectory_pass,
        "explanation_pass": explanation_pass,
        "joint_pass": trajectory_pass and explanation_pass,
        "evaluation_status": "complete",
    }


def canary(model_id: str, provider: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = Adapter(output_dir / "usage.jsonl")
    reply = adapter.complete(
        model=model_id,
        system="You are testing the DiscoverPhysics action protocol. Follow the requested XML format exactly.",
        messages=[{"role": "user", "content": "Return exactly <run_experiment>[]</run_experiment> and nothing else."}],
        max_tokens=2048,
    )
    recognized = bool(re.search(r"<run_experiment>.*?</run_experiment>", reply, re.DOTALL))
    (output_dir / "raw_reply.txt").write_text(reply)
    result = {
        "model_id": model_id,
        "provider": provider,
        "requested_reasoning": "high",
        "recognized_action": recognized,
        "reply_sha256": _hash_text(reply),
        "reply_chars": len(reply),
        "timestamp_utc": _utc_now(),
    }
    (output_dir / "canary.json").write_text(json.dumps(result, indent=2))
    return result


def _experiment_count(log: list[dict]) -> int:
    return sum(len(row.get("experiment_input") or []) for row in log if row.get("action") == "experiment")


def _hash_text(text: str | None) -> str | None:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def _safe(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
