from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

from dp_eval.adapters import Adapter, available_credentials, load_runtime_secrets
from dp_eval.metrics import aggregate
from dp_eval.sandbox import SandboxClient, worker_loop
from dp_eval.trial import canary, run_trial

UPSTREAM_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
PRIMARY_MODELS = [
    ("gpt-5.6-sol", "openai"),
    ("claude-opus-5", "anthropic"),
    ("gemini-3.7-flash", "google"),
    ("qwen/qwen3.5-397b-a17b", "openrouter"),
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="dp-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    worker = sub.add_parser("sandbox-worker")
    worker.add_argument("--ipc-root", default="/ipc")

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--offline", action="store_true")
    preflight.add_argument("--catalog-only", action="store_true")
    preflight.add_argument("--output", default=None)

    canaries = sub.add_parser("canaries")
    canaries.add_argument("--run-id", required=True)
    canaries.add_argument("--config")

    full_prep = sub.add_parser("validate-full-prep")
    full_prep.add_argument("--config", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--max-concurrency", type=int, default=1)

    agg = sub.add_parser("aggregate")
    agg.add_argument("run_dir")
    agg.add_argument("--label", default="Public-Pilot-4")

    args = parser.parse_args()
    if args.command == "sandbox-worker":
        worker_loop(args.ipc_root)
    elif args.command == "preflight":
        result = offline_preflight() if args.offline else catalog_preflight()
        if args.output:
            path = Path(args.output)
            path.mkdir(parents=True, exist_ok=True)
            (path / ("offline.json" if args.offline else "catalog.json")).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if not result.get("ok"):
            raise SystemExit(1)
    elif args.command == "canaries":
        result = run_canaries(args.run_id, Path(args.config) if args.config else None)
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            raise SystemExit(1)
    elif args.command == "validate-full-prep":
        result = validate_full_prep(Path(args.config))
        print(json.dumps(result, indent=2))
        if not result["template_valid"]:
            raise SystemExit(1)
    elif args.command == "run":
        result = run_matrix(Path(args.config), args.run_id, args.max_concurrency)
        print(json.dumps(result, indent=2))
        if result["failed"]:
            raise SystemExit(1)
    elif args.command == "aggregate":
        result = aggregate(Path(args.run_dir), args.label)
        print(json.dumps({"run_id": result["run_id"], "trial_count": result["trial_count"], "summary": str(Path(args.run_dir) / "aggregate" / "summary.md")}, indent=2))


def offline_preflight() -> dict[str, Any]:
    checks = {}
    try:
        from scienceagent.worlds import WORLDS
        paper_public_worlds = {
            "gravity", "yukawa", "fractional", "oscillator", "extra_dimensions",
            "coulomb_easy", "three_species", "dark_matter", "ether", "hubble", "circle",
        }
        checks["source_worlds"] = sorted(WORLDS)
        checks["paper_public_worlds"] = sorted(paper_public_worlds)
        checks["paper_public_world_count"] = len(paper_public_worlds)
        checks["extra_source_worlds"] = sorted(set(WORLDS) - paper_public_worlds)
        checks["paper_worlds_present"] = paper_public_worlds.issubset(WORLDS)
        checks["selected_worlds_present"] = all(w in WORLDS for w in ["gravity", "oscillator", "extra_dimensions", "dark_matter"])
    except Exception as exc:
        checks["import_error"] = f"{type(exc).__name__}: {exc}"

    try:
        sandbox = SandboxClient(timeout_s=10)
        checks["sandbox"] = sandbox.request({"action": "ping"})
    except Exception as exc:
        checks["sandbox_error"] = f"{type(exc).__name__}: {exc}"

    checks["upstream_commit"] = UPSTREAM_COMMIT
    checks["host_python_prefix"] = sys.prefix
    checks["ok"] = (
        checks.get("paper_public_world_count") == 11
        and checks.get("paper_worlds_present") is True
        and checks.get("selected_worlds_present") is True
        and checks.get("sandbox", {}).get("network_blocked") is True
        and checks.get("sandbox", {}).get("secret_visible") is False
        and checks.get("sandbox", {}).get("uid") not in (None, 0)
    )
    return checks


def catalog_preflight() -> dict[str, Any]:
    load_runtime_secrets()
    creds = available_credentials()
    result: dict[str, Any] = {"timestamp_utc": _utc_now(), "credentials_present": creds, "models": {}, "ok": False}

    # Metadata-only calls. Errors are captured without exposing credentials.
    try:
        from openai import OpenAI
        model = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).models.retrieve("gpt-5.6-sol")
        result["models"]["gpt-5.6-sol"] = {"available": True, "response_id": getattr(model, "id", None)}
    except Exception as exc:
        result["models"]["gpt-5.6-sol"] = _safe_error(exc)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        model = client.models.retrieve("claude-opus-5")
        result["models"]["claude-opus-5"] = {"available": True, "response_id": getattr(model, "id", None)}
        judge = client.models.retrieve("claude-opus-4-6")
        result["models"]["claude-opus-4-6"] = {"available": True, "response_id": getattr(judge, "id", None)}
    except Exception as exc:
        result["models"].setdefault("claude-opus-5", _safe_error(exc))
        result["models"].setdefault("claude-opus-4-6", _safe_error(exc))

    try:
        from google import genai
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        model = gemini_client.models.get(model="gemini-3.7-flash")
        result["models"]["gemini-3.7-flash"] = {"available": True, "response_id": getattr(model, "name", None)}
    except Exception as exc:
        result["models"]["gemini-3.7-flash"] = _safe_error(exc)

    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        response.raise_for_status()
        models = {row["id"]: row for row in response.json()["data"]}
        for model_id in ["qwen/qwen3.5-397b-a17b", "z-ai/glm-5.1", "nvidia/nemotron-3-ultra-550b-a55b"]:
            row = models.get(model_id)
            result["models"][model_id] = {
                "available": row is not None,
                "context_length": row.get("context_length") if row else None,
                "pricing": row.get("pricing") if row else None,
                "reasoning": row.get("reasoning") if row else None,
            }
        endpoints = requests.get("https://openrouter.ai/api/v1/models/qwen/qwen3.5-397b-a17b/endpoints", timeout=30)
        endpoints.raise_for_status()
        endpoint_rows = endpoints.json().get("data", {}).get("endpoints", [])
        result["openrouter_qwen_endpoints"] = [
            {k: row.get(k) for k in ("name", "provider_name", "context_length", "max_completion_tokens", "status")}
            for row in endpoint_rows
        ]
    except Exception as exc:
        result["models"].setdefault("qwen/qwen3.5-397b-a17b", _safe_error(exc))

    required = ["gpt-5.6-sol", "claude-opus-5", "claude-opus-4-6", "gemini-3.7-flash", "qwen/qwen3.5-397b-a17b"]
    result["ok"] = all(result["models"].get(mid, {}).get("available") for mid in required)
    return result


def validate_full_prep(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text())
    expected_public = {
        "gravity", "yukawa", "fractional", "oscillator", "extra_dimensions",
        "coulomb_easy", "three_species", "dark_matter", "ether", "hubble", "circle",
    }
    public = set(cfg.get("public_worlds") or [])
    private = cfg.get("private_worlds") or []
    lanes = cfg.get("lanes") or []
    arithmetic = int(cfg.get("expected_world_count", 0)) * len(cfg.get("seeds") or []) * len(lanes)
    checks = {
        "launch_authorized_is_false": cfg.get("launch_authorized") is False,
        "public_worlds_exact": public == expected_public,
        "private_worlds_deliberately_empty": private == [],
        "four_exact_lanes": [lane.get("model_id") for lane in lanes] == [
            "gpt-5.6-sol", "claude-opus-5", "gemini-3.7-flash", "qwen/qwen3.5-397b-a17b"
        ],
        "five_seeds": cfg.get("seeds") == [0, 1, 2, 3, 4],
        "expected_trial_arithmetic": arithmetic == 440 == cfg.get("expected_total_trials"),
        "max_tokens": cfg.get("max_tokens"),
    }
    return {
        "protocol_id": cfg.get("protocol_id"),
        "template_valid": all(value is True for key, value in checks.items() if key != "max_tokens") and checks["max_tokens"] == 16384,
        "ready_to_launch": False,
        "checks": checks,
        "blockers": [
            "full launch not authorized",
            "11 private world IDs absent pending approved gated access",
            "private ground-truth code must be isolated from generated-law filesystem",
            "Qwen/DeepInfra pilot retries unresolved",
        ],
    }


def run_canaries(run_id: str, config_path: Path | None = None) -> dict[str, Any]:
    artifact_root = Path(os.environ.get("DP_ARTIFACT_ROOT", "/artifacts"))
    if config_path:
        cfg = yaml.safe_load(config_path.read_text())
        models = [
            (lane["model_id"], lane["provider"], _endpoint_config(lane))
            for lane in cfg["lanes"]
        ]
    else:
        models = [(model_id, provider, None) for model_id, provider in PRIMARY_MODELS]
    rows = []
    for model_id, provider, endpoint_config in models:
        output = artifact_root / run_id / "raw" / "canaries" / _safe(model_id)
        try:
            row = canary(model_id, provider, output, endpoint_config)
            row["status"] = "completed" if row["recognized_action"] else "malformed"
        except Exception as exc:
            row = {"model_id": model_id, "provider": provider, "status": "failed", **_safe_error(exc)}
        rows.append(row)
    result = {"run_id": run_id, "timestamp_utc": _utc_now(), "canaries": rows, "ok": all(r["status"] == "completed" for r in rows)}
    out = artifact_root / run_id / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "canaries.json").write_text(json.dumps(result, indent=2))
    return result


def run_matrix(config_path: Path, run_id: str, max_concurrency: int) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text())
    for lane in cfg["lanes"]:
        endpoint_config = _endpoint_config(lane)
        if endpoint_config:
            Adapter(endpoint_config=endpoint_config).endpoint_canary()
            print(f"endpoint canary passed: {lane['label']}", flush=True)
    artifact_root = Path(os.environ.get("DP_ARTIFACT_ROOT", "/artifacts"))
    run_dir = artifact_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("protocol_sha256") != protocol_hash:
            raise ValueError("resume protocol hash differs from existing run manifest")
        if existing.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError("resume upstream commit differs from existing run manifest")
        if existing.get("openrouter_provider") != (os.environ.get("OPENROUTER_PROVIDER") or "auto"):
            raise ValueError("resume OpenRouter provider differs from existing run manifest")
    else:
        shutil.copyfile(config_path, run_dir / "protocol.yaml")
    specs = []
    # Interleave lanes so max_concurrency=4 starts one trial per provider,
    # rather than four simultaneous trials against the first provider.
    for world in cfg["worlds"]:
        for seed in cfg["seeds"]:
            for lane in cfg["lanes"]:
                specs.append({
                    "run_id": run_id,
                    "phase": cfg["phase"],
                    "lane": lane["label"],
                    "model_id": lane["model_id"],
                    "provider": lane["provider"],
                    "provider_backend": os.environ.get("OPENROUTER_PROVIDER") if lane["provider"] == "openrouter" else "native",
                    "world": world,
                    "seed": seed,
                    "max_rounds": cfg["max_rounds"],
                    "max_tokens": cfg["max_tokens"],
                    "noise_frac": cfg["noise_frac"],
                    "item_timeout_seconds": int(cfg.get("item_timeout_seconds", 7200)),
                    "endpoint_config": _endpoint_config(lane),
                })
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": _utc_now(),
        "protocol_id": cfg["protocol_id"],
        "protocol_sha256": protocol_hash,
        "upstream_commit": UPSTREAM_COMMIT,
        "trial_count": len(specs),
        "items": [
            {
                "item_key": _item_key(spec),
                "lane": spec["lane"],
                "model_id": spec["model_id"],
                "provider": spec["provider"],
                "world": spec["world"],
                "seed": spec["seed"],
            }
            for spec in specs
        ],
        "requested_reasoning": "high",
        "openrouter_provider": os.environ.get("OPENROUTER_PROVIDER") or "auto",
        "item_timeout_seconds": int(cfg.get("item_timeout_seconds", 7200)),
        "endpoint_models": [
            {
                key: lane.get(key)
                for key in (
                    "model_id", "base_url", "api_key_env", "reasoning_history",
                    "reasoning_effort", "request_parameters",
                )
            }
            for lane in cfg["lanes"]
            if _endpoint_config(lane)
        ],
        "raw_artifacts_in_git": False,
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    pending = []
    preserved_completed = 0
    preserved_failed_attempts = 0
    for spec in specs:
        trial_dir = _trial_dir(run_dir, spec)
        trial_path = trial_dir / "trial.json"
        if trial_path.exists():
            previous = json.loads(trial_path.read_text())
            if previous.get("status") == "completed":
                preserved_completed += 1
                continue
            preserved_failed_attempts += 1
            _preserve_invalid_attempt(trial_dir)
        elif trial_dir.exists() and any(trial_dir.iterdir()):
            _preserve_invalid_attempt(trial_dir)
        pending.append(spec)

    _run_isolated_processes(pending, max_concurrency, run_dir)
    final_rows = [json.loads(path.read_text()) for path in (run_dir / "raw").glob("*/*/seed-*/trial.json")]
    result = {
        "run_id": run_id,
        "planned": len(specs),
        "scheduled_this_invocation": len(pending),
        "preserved_completed": preserved_completed,
        "preserved_failed_attempts": preserved_failed_attempts,
        "finalized": len(final_rows),
        "completed": sum(r["status"] == "completed" for r in final_rows),
        "unresolved_missing": max(0, len(specs) - len(final_rows)),
        "failed": sum(r["status"] != "completed" for r in final_rows)
        + max(0, len(specs) - len(final_rows)),
        "run_dir": str(run_dir),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(result, indent=2))
    return result


def _run_isolated_processes(specs: list[dict[str, Any]], max_concurrency: int, run_dir: Path) -> None:
    """Run each trial in its own bounded process so failures cannot block other trials."""
    ctx = multiprocessing.get_context("spawn")
    waiting = iter(specs)
    active: list[tuple[multiprocessing.Process, dict[str, Any], float]] = []
    exhausted = False
    while active or not exhausted:
        while len(active) < max(1, max_concurrency) and not exhausted:
            try:
                spec = next(waiting)
            except StopIteration:
                exhausted = True
                break
            process = ctx.Process(target=run_trial, args=(spec,))
            process.start()
            active.append((process, spec, time.monotonic()))
        for process, spec, started in list(active):
            timed_out = (
                process.is_alive()
                and time.monotonic() - started >= spec["item_timeout_seconds"]
            )
            if process.is_alive() and not timed_out:
                continue
            if timed_out:
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
            process.join()
            if timed_out:
                _write_worker_exit(
                    run_dir, spec, process.exitcode,
                    error_type="ItemTimeout",
                    message=(
                        "trial exceeded configured item timeout of "
                        f"{spec['item_timeout_seconds']} seconds"
                    ),
                )
            elif process.exitcode != 0 and not (_trial_dir(run_dir, spec) / "trial.json").exists():
                _write_worker_exit(run_dir, spec, process.exitcode)
            active.remove((process, spec, started))
        if active:
            time.sleep(0.2)


def _trial_dir(run_dir: Path, spec: dict[str, Any]) -> Path:
    return run_dir / "raw" / _safe(spec["lane"]) / spec["world"] / f"seed-{int(spec['seed'])}"


def _retryable_infrastructure_failure(row: dict[str, Any]) -> bool:
    error = row.get("error") or {}
    message = str(error.get("message") or "")
    return (
        error.get("type") in {
            "WorkerExit", "ItemTimeout", "ChunkedEncodingError", "ConnectionError"
        }
        or "Provider request failed after retries" in message
        or "RemoteDisconnected" in message
    )


def _preserve_invalid_attempt(trial_dir: Path) -> None:
    target = trial_dir / "invalid-infrastructure-attempt"
    suffix = 1
    while target.exists():
        suffix += 1
        target = trial_dir / f"invalid-infrastructure-attempt-{suffix}"
    target.mkdir(parents=True)
    for child in list(trial_dir.iterdir()):
        if child == target or child.name.startswith("invalid-infrastructure-attempt"):
            continue
        child.rename(target / child.name)


def _write_worker_exit(
    run_dir: Path,
    spec: dict[str, Any],
    exitcode: int | None,
    error_type: str = "WorkerExit",
    message: str | None = None,
) -> None:
    trial_dir = _trial_dir(run_dir, spec)
    trial_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1, "run_id": spec["run_id"], "phase": spec["phase"],
        "item_key": _item_key(spec),
        "lane": spec["lane"], "model_id": spec["model_id"], "provider": spec["provider"],
        "provider_backend": spec.get("provider_backend"), "requested_reasoning": "high",
        "world": spec["world"], "world_visibility": "public", "seed": int(spec["seed"]),
        "status": "failed", "max_rounds": spec["max_rounds"], "max_tokens": spec["max_tokens"],
        "item_timeout_seconds": spec["item_timeout_seconds"],
        "error": {
            "type": error_type,
            "message": message or f"isolated trial process exited with code {exitcode}",
        },
    }
    (trial_dir / "trial.json").write_text(json.dumps(row, indent=2))


def _item_key(spec: dict[str, Any]) -> str:
    return f"{_safe(spec['lane'])}/{spec['world']}/seed-{int(spec['seed'])}"


def _endpoint_config(lane: dict[str, Any]) -> dict[str, Any] | None:
    if lane.get("provider") in {"openai_compatible", "vllm"}:
        return lane
    return None


def _safe_error(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "[REDACTED]")
    return {"available": False, "error_type": type(exc).__name__, "error": message[:500]}


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "_.-" else "_" for c in value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
