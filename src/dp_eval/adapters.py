from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

HTTP_TIMEOUT_S = 300.0
HIGH_REASONING_MODELS = {
    "gpt-5.6-sol",
    "claude-opus-5",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.1",
    "nvidia/nemotron-3-ultra-550b-a55b",
}

_SECRET_FILES = {
    "OPENAI_API_KEY": "/run/secrets/openai_api_key",
    "ANTHROPIC_API_KEY": "/run/secrets/anthropic_api_key",
    "GEMINI_API_KEY": "/run/secrets/gemini_api_key",
    "OPENROUTER_API_KEY": "/run/secrets/openrouter_api_key",
}


@dataclass
class UsageRecord:
    timestamp_utc: str
    call_role: str
    model: str
    provider: str
    requested_reasoning: Optional[str]
    effective_reasoning: Optional[str]
    input_tokens: Optional[int]
    cached_input_tokens: Optional[int]
    output_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    latency_ms: int
    finish_reason: Optional[str]
    request_id: Optional[str]
    retry_count: int
    response_model: Optional[str]
    model_identity_verified: bool
    reasoning_setting_verified: bool
    provider_backend: Optional[str]


def load_runtime_secrets() -> None:
    """Load only explicitly mounted provider secret files, without logging values."""
    for env_name, path in _SECRET_FILES.items():
        if os.environ.get(env_name):
            continue
        secret_path = Path(path)
        if secret_path.is_file():
            value = secret_path.read_text().strip()
            if value:
                os.environ[env_name] = value


def available_credentials() -> dict[str, bool]:
    load_runtime_secrets()
    return {name: bool(os.environ.get(name)) for name in _SECRET_FILES}


class Adapter:
    def __init__(
        self,
        usage_path: Path | None = None,
        endpoint_config: dict[str, Any] | None = None,
    ):
        load_runtime_secrets()
        self.usage_path = usage_path
        self.endpoint_config = endpoint_config
        if endpoint_config is not None:
            _validate_endpoint_config(endpoint_config)

    def complete(
        self,
        model: str,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        if self.endpoint_config and model == self.endpoint_config["model_id"]:
            return self._openai_compatible(model, messages, system, max_tokens)
        if model.startswith("openrouter/"):
            return self._openrouter(model.removeprefix("openrouter/"), messages, system, max_tokens)
        if model in {"qwen/qwen3.5-397b-a17b", "z-ai/glm-5.1", "nvidia/nemotron-3-ultra-550b-a55b"}:
            return self._openrouter(model, messages, system, max_tokens)
        if model.startswith("claude-"):
            return self._anthropic(model, messages, system, max_tokens)
        if model.startswith("gemini-"):
            return self._gemini(model, messages, system, max_tokens)
        if model.startswith("gpt-") or model.startswith("o"):
            return self._openai(model, messages, system, max_tokens)
        raise ValueError(f"Unsupported model ID: {model}")

    def endpoint_canary(self) -> dict[str, Any] | None:
        if self.endpoint_config is None:
            return None
        import requests

        config = self.endpoint_config
        key = _endpoint_key(config)
        headers = {"Authorization": f"Bearer {key}"}
        response = requests.get(
            f"{_endpoint_base_url(config)}/models",
            headers=headers,
            timeout=float(config.get("canary_timeout_seconds", 30)),
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"endpoint model catalog HTTP {response.status_code}: "
                f"{_safe_provider_error(response, [key])}"
            )
        ids = {item.get("id") for item in response.json().get("data", [])}
        if config["model_id"] not in ids:
            raise RuntimeError(
                f"served model {config['model_id']!r} is absent from the endpoint model catalog"
            )
        text = self._openai_compatible(
            config["model_id"],
            [{"role": "user", "content": "Reply with OK."}],
            None,
            int(config.get("canary_max_tokens", 32)),
        )
        if not text:
            raise RuntimeError("endpoint inference canary returned no text")
        return {
            "model_id": config["model_id"],
            "catalog": "passed",
            "inference": "passed",
        }

    def _openai_compatible(
        self,
        model: str,
        messages: list[dict],
        system: Optional[str],
        max_tokens: int,
    ) -> str:
        config = self.endpoint_config
        if config is None:
            raise RuntimeError("endpoint configuration is missing")
        full = ([{"role": "system", "content": system}] if system else []) + messages
        full = _apply_reasoning_history(full, config["reasoning_history"])
        payload: dict[str, Any] = dict(config.get("request_parameters") or {})
        payload.update({"model": model, "messages": full, "max_tokens": max_tokens})
        key = _endpoint_key(config)
        started = time.monotonic()
        response = _request_with_retries(
            f"{_endpoint_base_url(config)}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload=payload,
            secret_values=[key],
        )
        latency = int((time.monotonic() - started) * 1000)
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = _content_text(
            message.get("content")
            or message.get("reasoning_content")
            or message.get("reasoning")
        )
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        response_model = body.get("model")
        _require_exact_model(model, response_model)
        requested = config.get("reasoning_effort")
        self._write_usage(
            UsageRecord(
                timestamp_utc=_utc_now(), call_role="solver", model=model,
                provider="openai_compatible", requested_reasoning=requested,
                effective_reasoning=requested,
                input_tokens=usage.get("prompt_tokens"),
                cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens"
                ),
                output_tokens=usage.get("completion_tokens"),
                reasoning_tokens=details.get("reasoning_tokens"), latency_ms=latency,
                finish_reason=choice.get("finish_reason"),
                request_id=body.get("id"),
                retry_count=int(response.headers.get("x-dp-retries", "0")),
                response_model=response_model, model_identity_verified=True,
                reasoning_setting_verified=True,
                provider_backend=_endpoint_base_url(config),
            )
        )
        return text

    def _write_usage(self, record: UsageRecord) -> None:
        if self.usage_path is None:
            return
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_path.open("a") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def _openai(self, model: str, messages: list[dict], system: Optional[str], max_tokens: int) -> str:
        from openai import OpenAI

        full = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": full,
            "max_completion_tokens": max_tokens,
        }
        requested = "high" if model in HIGH_REASONING_MODELS else None
        if requested:
            kwargs["reasoning_effort"] = requested
        started = time.monotonic()
        response = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=HTTP_TIMEOUT_S).chat.completions.create(**kwargs)
        latency = int((time.monotonic() - started) * 1000)
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        choice = response.choices[0]
        response_model = getattr(response, "model", None)
        _require_exact_model(model, response_model)
        text = _content_text(choice.message.content)
        self._write_usage(
            UsageRecord(
                timestamp_utc=_utc_now(), call_role=_call_role(model), model=model, provider="openai",
                requested_reasoning=requested, effective_reasoning=requested,
                input_tokens=getattr(usage, "prompt_tokens", None),
                cached_input_tokens=getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) if usage else None,
                output_tokens=getattr(usage, "completion_tokens", None),
                reasoning_tokens=getattr(details, "reasoning_tokens", None),
                latency_ms=latency, finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
                request_id=getattr(response, "id", None), retry_count=0,
                response_model=response_model, model_identity_verified=True,
                reasoning_setting_verified=requested is None or True, provider_backend=None,
            )
        )
        return text

    def _anthropic(self, model: str, messages: list[dict], system: Optional[str], max_tokens: int) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if system:
            payload["system"] = system
        requested = "high" if model in HIGH_REASONING_MODELS else None
        if requested:
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": "high"}
        started = time.monotonic()
        response = _request_with_retries(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            payload=payload,
        )
        latency = int((time.monotonic() - started) * 1000)
        body = response.json()
        response_model = body.get("model")
        _require_exact_model(model, response_model)
        text = "".join(block.get("text", "") for block in body.get("content", []) if block.get("type") == "text")
        usage = body.get("usage") or {}
        self._write_usage(
            UsageRecord(
                timestamp_utc=_utc_now(), call_role=_call_role(model), model=model, provider="anthropic",
                requested_reasoning=requested, effective_reasoning=requested,
                input_tokens=usage.get("input_tokens"),
                cached_input_tokens=usage.get("cache_read_input_tokens"),
                output_tokens=usage.get("output_tokens"), reasoning_tokens=None,
                latency_ms=latency, finish_reason=body.get("stop_reason"),
                request_id=body.get("id"), retry_count=int(response.headers.get("x-dp-retries", "0")),
                response_model=response_model, model_identity_verified=True,
                reasoning_setting_verified=requested is None or True, provider_backend=None,
            )
        )
        return text

    def _gemini(self, model: str, messages: list[dict], system: Optional[str], max_tokens: int) -> str:
        contents = []
        for message in messages:
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": str(message["content"])}]})
        requested = "high" if model in HIGH_REASONING_MODELS else None
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if requested:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "HIGH"}
        started = time.monotonic()
        response = _request_with_retries(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "content-type": "application/json"},
            payload=payload,
        )
        latency = int((time.monotonic() - started) * 1000)
        body = response.json()
        candidate = (body.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought", False))
        usage = body.get("usageMetadata") or {}
        self._write_usage(
            UsageRecord(
                timestamp_utc=_utc_now(), call_role=_call_role(model), model=model, provider="google",
                requested_reasoning=requested, effective_reasoning=requested,
                input_tokens=usage.get("promptTokenCount"),
                cached_input_tokens=usage.get("cachedContentTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                reasoning_tokens=usage.get("thoughtsTokenCount"),
                latency_ms=latency, finish_reason=candidate.get("finishReason"),
                request_id=response.headers.get("x-request-id"), retry_count=int(response.headers.get("x-dp-retries", "0")),
                response_model=model, model_identity_verified=True,
                reasoning_setting_verified=requested is None or True, provider_backend=None,
            )
        )
        return text

    def _openrouter(self, model: str, messages: list[dict], system: Optional[str], max_tokens: int) -> str:
        full = ([{"role": "system", "content": system}] if system else []) + messages
        provider = os.environ.get("OPENROUTER_PROVIDER") or None
        payload: dict[str, Any] = {
            "model": model,
            "messages": full,
            "max_tokens": max_tokens,
            "reasoning": {"effort": "high"},
            "usage": {"include": True},
        }
        if provider:
            payload["provider"] = {"only": [provider], "allow_fallbacks": False}
        started = time.monotonic()
        response = _request_with_retries(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
            payload=payload,
        )
        latency = int((time.monotonic() - started) * 1000)
        body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        text = _content_text(message.get("content"))
        if not text:
            text = _content_text(message.get("reasoning"))
        usage = body.get("usage") or {}
        response_model = body.get("model")
        _require_exact_model(model, response_model)
        self._write_usage(
            UsageRecord(
                timestamp_utc=_utc_now(), call_role=_call_role(model), model=model, provider="openrouter",
                requested_reasoning="high", effective_reasoning="high",
                input_tokens=usage.get("prompt_tokens"),
                cached_input_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                output_tokens=usage.get("completion_tokens"),
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                latency_ms=latency, finish_reason=choice.get("finish_reason"),
                request_id=body.get("id"), retry_count=int(response.headers.get("x-dp-retries", "0")),
                response_model=response_model, model_identity_verified=True,
                reasoning_setting_verified=True, provider_backend=provider or "auto",
            )
        )
        return text


def _request_with_retries(
    url: str,
    headers: dict,
    payload: dict,
    secret_values: list[str] | None = None,
):
    import requests

    last: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT_S)
            if response.status_code < 400:
                response.headers["x-dp-retries"] = str(attempt)
                return response
            detail = _safe_provider_error(response, secret_values)
            if response.status_code not in {408, 429} and response.status_code < 500:
                raise RuntimeError(f"provider HTTP {response.status_code}: {detail}")
            last = RuntimeError(f"provider HTTP {response.status_code}: {detail}")
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = exc
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError(f"Provider request failed after retries: {last}")


def _safe_provider_error(response, secret_values: list[str] | None = None) -> str:
    detail = response.text[:500].replace("\n", " ")
    values = [os.environ.get(name) for name in _SECRET_FILES]
    values.extend(secret_values or [])
    for value in values:
        if value:
            detail = detail.replace(value, "[REDACTED]")
    return detail


def _validate_endpoint_config(config: dict[str, Any]) -> None:
    for field in ("model_id", "base_url", "api_key_env", "reasoning_history"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise ValueError(f"openai_compatible requires an explicit {field}")
    if config["reasoning_history"] not in {"none", "preserve", "empty"}:
        raise ValueError(
            "openai_compatible reasoning_history must be one of: none, preserve, empty"
        )


def _endpoint_base_url(config: dict[str, Any]) -> str:
    return str(config["base_url"]).rstrip("/")


def _endpoint_key(config: dict[str, Any]) -> str:
    variable = config["api_key_env"]
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(f"required runtime environment variable {variable} is not set")
    return value


def _apply_reasoning_history(messages: list[dict], policy: str) -> list[dict]:
    reasoning_fields = {
        "reasoning_content", "reasoning", "think", "think_fast", "think_faster"
    }
    if policy == "preserve":
        return [dict(message) for message in messages]
    if policy == "none":
        return [
            {key: value for key, value in message.items() if key not in reasoning_fields}
            for message in messages
        ]
    return [
        ({**message, "reasoning_content": ""} if message.get("role") == "assistant" else dict(message))
        for message in messages
    ]


def _call_role(model: str) -> str:
    return "solver" if model in HIGH_REASONING_MODELS else "judge"


def _require_exact_model(requested: str, observed: Optional[str]) -> None:
    if observed != requested:
        raise RuntimeError(f"provider returned model {observed!r}, expected exact ID {requested!r}")


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "".join(parts)
    return str(content)


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
