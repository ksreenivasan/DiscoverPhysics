from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


class SandboxClient:
    def __init__(self, ipc_root: str | Path | None = None, timeout_s: float = 900.0):
        self.root = Path(ipc_root or os.environ.get("DP_IPC_ROOT", "/ipc"))
        self.in_dir = self.root / "in"
        self.out_dir = self.root / "out"
        self.in_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        final = self.in_dir / f"{job_id}.json"
        temp = self.in_dir / f".{job_id}.tmp"
        temp.write_text(json.dumps(payload, default=_json_default))
        temp.replace(final)
        response_path = self.out_dir / f"{job_id}.json"
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if response_path.exists():
                result = json.loads(response_path.read_text())
                response_path.unlink(missing_ok=True)
                if not result.get("ok"):
                    raise RuntimeError(f"sandbox {payload.get('action')} failed: {result.get('error')}")
                return result["result"]
            time.sleep(0.1)
        raise TimeoutError(f"sandbox job {job_id} exceeded {self.timeout_s:g}s")


def worker_loop(ipc_root: str | Path = "/ipc", poll_s: float = 0.1) -> None:
    root = Path(ipc_root)
    in_dir = root / "in"
    out_dir = root / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("law sandbox ready", flush=True)
    while True:
        jobs = sorted(in_dir.glob("*.json"))
        if not jobs:
            time.sleep(poll_s)
            continue
        for job_path in jobs:
            claimed = job_path.with_suffix(".working")
            try:
                job_path.replace(claimed)
            except FileNotFoundError:
                continue
            try:
                payload = json.loads(claimed.read_text())
                result = _handle(payload)
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
            out_path = out_dir / f"{claimed.stem}.json"
            tmp = out_dir / f".{claimed.stem}.tmp"
            tmp.write_text(json.dumps(response, default=_json_default))
            tmp.replace(out_path)
            claimed.unlink(missing_ok=True)


def _handle(payload: dict[str, Any]) -> dict[str, Any]:
    action = payload["action"]
    if action == "ping":
        import socket

        secret_names = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"
        ]
        secret_visible = any(os.environ.get(name) for name in secret_names)
        network_blocked = False
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=1).close()
        except OSError:
            network_blocked = True
        return {
            "network_blocked": network_blocked,
            "secret_visible": secret_visible,
            "uid": os.getuid(),
        }
    if action == "fit_law":
        from scienceagent.mse_fitting import fit_law

        return fit_law(
            law_source=payload["law_source"],
            world=payload["world"],
            csv_path=Path(payload["csv_path"]),
            run_id=payload["run_id"],
        )
    if action == "evaluate":
        return _evaluate(payload)
    raise ValueError(f"unknown sandbox action: {action}")


def _evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    from scienceagent.evaluator import (
        CircleEvaluator,
        DarkMatterEvaluator,
        EtherEvaluator,
        Evaluator,
        HubbleEvaluator,
        ThreeSpeciesEvaluator,
    )
    from scienceagent.worlds import get_world

    world_name = payload["world"]
    world = get_world(
        world_name,
        engine=payload.get("engine", "nbody"),
        noise_std=float(payload.get("noise_std", 0.0)),
        noise_seed=payload.get("noise_seed"),
    )
    executor = world["executor"]
    if world_name == "circle":
        evaluator = CircleEvaluator(executor)
    elif world_name == "three_species":
        evaluator = ThreeSpeciesEvaluator(executor)
    elif world_name == "dark_matter":
        evaluator = DarkMatterEvaluator(executor)
    elif world_name == "ether":
        evaluator = EtherEvaluator(executor)
    elif world_name == "hubble":
        evaluator = HubbleEvaluator(executor)
    elif world.get("evaluator_class") is not None:
        evaluator = world["evaluator_class"](executor)
    else:
        evaluator = Evaluator(executor)

    fit_worlds = {
        "gravity", "yukawa", "fractional", "diffusion", "wave",
        "oscillator", "extra_dimensions", "circle", "ether", "hubble",
    }
    kwargs: dict[str, Any] = {"verbose": False}
    if world_name in fit_worlds:
        kwargs["training_trajectories"] = payload.get("training_trajectories") or []
    return evaluator.evaluate(payload["law_source"], **kwargs)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
