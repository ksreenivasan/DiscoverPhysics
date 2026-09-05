# DiscoverPhysics implementation and tool audit

> **Public data only:** all completed runs use only worlds published in the canonical GitHub repository. The gated Hugging Face suite has never been accessed or downloaded, and no private world names, laws, rubrics, generated laws, or raw traces are tracked on this branch.

## Canonical provenance

| Artifact | Canonical source | Revision used or checked | How it is used |
|---|---|---|---|
| Paper | [arXiv:2605.26087v1](https://arxiv.org/abs/2605.26087v1) | v1 | Defines 22 worlds (11 public/11 private), five attempts per world, the ScienceAgent experiment loop, main noise/round condition, normalized trajectory MSE, explanation judge, and Pass@k interpretation. |
| Public implementation | [SampsonML/DiscoverPhysics](https://github.com/SampsonML/DiscoverPhysics) | [`33b7fa9df96de9c35744efd181ca7e5a8dd60ad5`](https://github.com/SampsonML/DiscoverPhysics/tree/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5) | Reused simulator, public world definitions, prompts, ScienceAgent, fitting code, evaluators, rubrics, tests, and analysis constants. |
| License | [MIT license at the pinned commit](https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/LICENSE) | pinned commit | Governs the public source copied into the image. |
| Official project/leaderboard | [DiscoverPhysics leaderboard](https://sampsonml.github.io/DiscoverPhysicsLeaderboard/) | checked 2026-08-25 | Confirms the official joint threshold: normalized MSE `<0.1` and explanation score `>=0.9`. |
| Protected suite | [mattWiemann/DiscoverPhysics](https://huggingface.co/mattWiemann/DiscoverPhysics) | not accessed | Manual-gated source described by the official project. It has never been downloaded or used by this branch. |

The canonical repository is research source, not a versioned PyPI benchmark release. Its `PhysicsSchool/setup.py` and `ScienceAgent/setup.py` declare local packages `physchool` and `scienceagent` at version `0.0.1`; there are no upstream tags, releases, lockfile, or container image. This repository therefore pins the Git commit rather than a package version.

This feature branch starts directly from the pinned upstream commit. `EVAL_BASE_REVISION` records the canonical Git URL/SHA, and the Docker image copies the fork's canonical `PhysicsSchool/` and `ScienceAgent/` trees in-place rather than using a nested checkout. No private source is committed.

## Canonical code reused without changing task semantics

- `PhysicsSchool/physchool/worlds/`: N-body/field simulation and force laws.
- `PhysicsSchool/prompts/`: the model-facing interactive instructions and world-topology schemas.
- `ScienceAgent/scienceagent/agent.py`: XML action loop (`run_experiment`, optional `run_mse_fit`, `final_law`, `explanation`).
- `ScienceAgent/scienceagent/worlds.py`: public world definitions, optimal explanations, and rubrics.
- `ScienceAgent/scienceagent/executor.py`: conversion of model experiment JSON into simulated trajectories.
- `ScienceAgent/scienceagent/mse_fitting.py`: optional fit of up to five submitted parameters against observed trajectories.
- `ScienceAgent/scienceagent/evaluator.py`: held-out trajectory evaluation and `claude-opus-4-6` explanation judge prompt/parser.
- `scripts/run_benchmark.py`: per-world variance constants used for normalized MSE.
- Upstream test suites: 184 passed and 1 skipped in the pinned container environment.

The pinned `WORLDS` registry contains an additional `wave` source entry. The paper's declared public eleven are explicitly pinned in our protocol; `wave` is not silently added to Public-Pilot-4.

## Locally written implementation

| Local path | Purpose | Why it is local rather than canonical |
|---|---|---|
| `src/dp_eval/adapters.py` | Current OpenAI, Anthropic, Gemini, and OpenRouter calls; high reasoning; exact returned-model checks; usage metadata; bounded transport retry. | Upstream lacks Gemini, current effort controls, exact identity checks, and sufficient usage/routing records. |
| `src/dp_eval/trial.py` | Instantiates the canonical world/ScienceAgent, records artifacts, delegates generated-law work, invokes the canonical judge, and recomputes the official 0.1/0.9 joint pass. | Upstream CLI executes generated Python in the credentialed process and its checked aggregation threshold is 0.75 rather than the paper/leaderboard 0.9. |
| `src/dp_eval/sandbox.py` | Narrow file-queue worker for fitting/evaluating generated laws with no secrets or network. | Host/credential isolation is absent upstream. |
| `src/dp_eval/cli.py` | Offline/catalog preflight, fixed matrix expansion, independent trial-process supervision, exact manifest validation, and selective infrastructure resume. | Operational harness needed for a bounded multi-provider campaign. |
| `src/dp_eval/metrics.py` | Internal Public-Pilot-4 aggregation and exact combinatorial subset Pass@1/3/5. | Prevents a four-world subset from being mislabeled as the full benchmark. |
| `Dockerfile`, `compose.yaml` | Reproducible dependencies and proportional two-container isolation. | Upstream has no container/lockfile. |
| `configs/*.yaml` | Explicit user-approved model/provider/scaffold conditions. | The upstream example has only two seeds and stale aggregation defaults. |

Pinned dependency versions in `requirements.txt` are local environment choices, not official benchmark pins. The base image used for the pilot is `python:3.11-slim-bookworm` at digest `sha256:fe556eaffd0d96abd8685b83bb8a69addd0dc730fa193669a7df0d49ddcfac02`, running Linux ARM64.

### OpenAI-compatible endpoint lanes

An arbitrary OpenAI-compatible or vLLM deployment is configured as one lane; endpoint coordinates are never embedded in the adapter:

```yaml
lanes:
  - label: served-model
    provider: openai_compatible
    model_id: exact-served-model-id
    base_url: https://model-host.example/v1
    api_key_env: SERVED_MODEL_API_KEY
    reasoning_history: empty
    request_parameters: {}
```

`base_url`, exact `model_id`, `api_key_env`, and `reasoning_history` (`none`, `preserve`, or `empty`) are mandatory. The named variable must hold either the real key or an explicit dummy token. The `empty` history policy carries forward the proven vLLM compatibility behavior by adding an empty `reasoning_content` to replayed assistant turns; `none` strips reasoning fields and `preserve` passes them through. Before `run` starts any trial, it requires the exact ID from `GET <base_url>/models` and a small successful Chat Completions inference.

Native `gemini-3.8-flash` requests high thinking and sends only `maxOutputTokens` plus `thinkingConfig`; rejected legacy sampling parameters are not sent. Older model behavior is unchanged.

Each protocol sets `item_timeout_seconds`. Every deterministic `lane/world/seed` trial runs in a separate bounded process, writes its result immediately, and is skipped only after a completed artifact exists. Exceptions and timeouts become explicit failed trial artifacts; retryable failures are preserved and retried on resume. Aggregation reconciles artifacts against manifest item keys and keeps failed or missing items in the denominator.

## What tools the evaluated model actually has

### Available to the solver

1. **Interactive experiment API — benchmark required.** The model emits a `<run_experiment>` XML block containing world-specific JSON initial conditions and measurement times. The canonical executor runs the hidden world and returns noisy positions/velocities in `<experiment_output>`. This is the central task interaction, not an optional external tool.
2. **Optional MSE fitting API — canonical benchmark scaffold.** The model may emit `<run_mse_fit>` with candidate Python plus an optional `fit_parameters()` declaration. Canonical Scipy fitting uses only trajectories already collected in that attempt and returns losses/fitted values. It does not reveal held-out trajectories or a world definition.
3. **Final executable law — benchmark required output.** The model submits Python inside `<final_law>` plus a prose `<explanation>`. The code is executed only for fitting/scoring in the isolated law worker.

### Not available to the solver

- no web search or browser;
- no shell/terminal tool;
- no arbitrary notebook or general-purpose code interpreter during reasoning;
- no direct filesystem API;
- no direct network access;
- no Docker API/socket;
- no provider credentials;
- no held-out trajectories, optimal explanation, or rubric;
- no critic/supervisor model (`critic: off`);
- no random-experiment replacement (`random_experiments: off`).

The provider itself receives normal conversation history, not hosted provider tools. OpenAI/Gemini/Anthropic tool features are not enabled. OpenRouter is pinned to DeepInfra for Qwen with provider fallback disabled.

## Environment and trust boundaries

### Runner container

The runner contains the public simulator, ScienceAgent, provider adapters, and explanation judge client. It receives the four explicitly authorized key files from `~/secrets_and_keys` as runtime Docker secrets. It has ordinary outbound bridge networking only because model and judge API calls require it. It has no host networking, privileged mode, Docker socket, or broad writable host mount. Its only writable host bind is the dedicated external run-artifact directory.

The hidden world is simulated in the runner. Model experiment JSON is data, not executed code. Before a generated candidate law would be compiled, the runner submits a narrow job to the law sandbox.

### Generated-law sandbox container

The law sandbox is non-root UID 10001, has `network_mode: none`, a read-only root filesystem, all Linux capabilities dropped, `no-new-privileges`, bounded PIDs/memory/CPU, no Docker socket, and no provider secret mounts. It receives law source and the minimum numeric fit/evaluation job through a dedicated named volume. This boundary is necessary because the upstream evaluator uses Python `exec` on model output.

For public worlds, the current sandbox image contains the public canonical source. That is harmless for contamination because those definitions are already public. It is **not sufficient for private-world validity**: putting gated definitions in the same filesystem as arbitrary generated code would make them readable. A full private launch is blocked until private ground-truth simulation is separated from the minimal law-execution filesystem (see the full-run runbook).

### Judge and scoring path

1. The sandbox executes the submitted law on canonical held-out cases and returns raw trajectory errors.
2. The runner divides mean squared position error by the pinned per-world ground-truth variance.
3. Separately, native Anthropic `claude-opus-4-6` receives the model's prose explanation, the hidden optimal explanation, and the world-specific rubric. It returns an integer 0–10 score, normalized to 0–1.
4. A trial jointly passes only if normalized MSE `<0.1` and explanation score `>=0.9`.
5. Five attempt positions support exact without-replacement Pass@1/3/5. A four-world public pilot is labeled `Public-Pilot-4`; it is not an official full score.

The judge is evaluation infrastructure, not a tool available to the solver. Solver and judge usage records are explicitly separated with `call_role`.

## Benchmark requirements versus implementation choices

| Condition | Benchmark-required/canonical | Local/user-approved choice |
|---|---|---|
| Adaptive simulated experiments | Required task behavior | Reused unchanged. |
| ScienceAgent XML prompt/action loop | Official scaffold used by the paper | Reused; local adapter only transports messages. |
| Up to 16 rounds | Paper condition | Fixed at 16. |
| Five attempts/world | Paper condition | Seeds/attempt positions 0–4. |
| Noise | Paper main condition | `noise_frac=0.05`, clean held-out evaluation. |
| N-body/Yoshida-4 | Canonical released static-world backend | Fixed for all selected public worlds. |
| Mid-round fit | Canonical optional tool | Enabled; no critic/random replacement. |
| Trajectory and explanation scoring | Required | Canonical evaluator/judge, independently audited 0.1/0.9 aggregation. |
| Solver model/provider | Not benchmark-fixed | Four exact requested IDs; native providers except Qwen via OpenRouter/DeepInfra. |
| Reasoning effort | Provider-specific | High for every solver, requested and logged. |
| Output cap | Evaluation condition | 16,384 after high-thinking Gemini smoke validation. |
| Provider retry | Operational choice | Two retries only for transient transport/408/429/5xx; invalid attempts are preserved. |
| Concurrency | Operational choice | Four supervised processes, interleaved one per provider initially. |
| Container sandbox | Safety choice | Added without changing model-visible actions or numeric scorer inputs. |
| Artifact layout | Reproducibility choice | Raw/large/protected data external to Git; sanitized reports only in Git. |

## Protected-data gate

The Hugging Face repository is manually gated. Its terms prohibit redistribution/publication of private definitions/laws/rubrics and their use for training, and require citation and honest reproducible reporting. Full-run preparation must pin the approved snapshot, keep it outside Git, and produce a sanitized distributable report. No private access or download has occurred in this implementation session.
