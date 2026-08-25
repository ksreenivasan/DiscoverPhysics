# DiscoverPhysics full-run preparation runbook

## Status

> **Public data only:** current runs use only worlds published in this GitHub repository. Gated artifacts have never been downloaded. This runbook contains public links and access conditions only, not protected world metadata.

**Prepared only; launch is not authorized.** `configs/full.template.yaml` deliberately has `launch_authorized: false` and an empty `private_worlds` list. It cannot be passed to the current `run` command.

The intended official-size matrix is:

```text
22 worlds × 5 seeds × 4 solver lanes = 440 trials
110 trials per solver lane
```

## Gates before implementation or launch

1. **Current pilot resolution.** Public-Pilot-4 is paused with 80 current identities: 69 completed and 11 exact Qwen/DeepInfra infrastructure failures. Preserve all current and archived attempts. Decide whether/when to selectively retry those exact identities; do not change provider or model.
2. **Explicit launch authorization.** A full run remains outside the current authorization.
3. **Gated access.** An authorized individual must accept the terms for [mattWiemann/DiscoverPhysics](https://huggingface.co/mattWiemann/DiscoverPhysics). Pin the approved commit SHA and store the checkout outside Git.
4. **Canonical snapshot comparison.** Compare public simulator/prompt/evaluator files in the gated snapshot against GitHub commit `33b7fa9...`. Use one coherent snapshot or document exact compatible patches; do not mix revisions casually.
5. **Private-world manifest.** Load the 11 private world IDs at runtime from the protected snapshot. Do not write their names, definitions, laws, or rubrics into committed config, logs intended for publication, or reports.
6. **Private execution boundary.** Do not place protected definitions in the same filesystem namespace as model-generated Python. Add the smallest separation that lets a trusted ground-truth process produce numeric held-out jobs while the existing no-network law sandbox receives only call inputs/expected numeric trajectories. Validate with a fixture private world before real access is used.
7. **Judge availability.** Verify exact native `claude-opus-4-6`; do not silently replace it.
8. **Model/provider availability.** Verify exact requested IDs and high reasoning. Qwen must remain `qwen/qwen3.5-397b-a17b` through the approved OpenRouter provider condition. No fallback model/provider without a new explicit decision.
9. **Artifact policy.** Select a protected external artifact root, access permissions, retention period, backup policy, and sanitized-export review owner.
10. **Observed estimate review.** Approve runtime/spend after reviewing the estimates below; spend is recorded rather than used as an automatic stop.

## Smallest implementation delta after approval

Do not build a general platform. Extend the present two-container design only as required for private confidentiality:

- trusted runner/ground-truth component: gated world definitions, simulator, rubrics, provider/judge access;
- existing generated-law sandbox: no gated source, secrets, or network; receives a narrow serialized law-evaluation job;
- external protected artifacts: raw traces and private metadata;
- committed sanitized report: aggregate metrics and public-safe provenance only.

The current public evaluator reconstructs a world inside the law sandbox. For private worlds, replace that path with a generic job containing the submitted function signature/call arguments and numeric ground-truth targets. The ground-truth component must perform private simulation; the sandbox must perform only generated-law calls and return predictions/errors. Parameter-fitting jobs receive only observed trajectories already available to the solver.

## Offline preparation sequence

These commands are illustrative and remain no-cost/offline until a separate authorization changes `launch_authorized` and supplies gated data.

```bash
# Validate the non-runnable template and trial arithmetic with networking disabled.
docker run --rm --network none \
  --entrypoint dp-eval discoverphysics-public-harness:local \
  validate-full-prep --config /app/configs/full.template.yaml

# Re-run local and canonical tests with no provider calls.
docker compose run --rm --no-deps --entrypoint python runner \
  -m pytest -q -p no:cacheprovider /app/tests
docker compose run --rm --no-deps --entrypoint python runner \
  -m pytest -q -p no:cacheprovider PhysicsSchool/tests ScienceAgent/tests
```

After gated access and explicit authorization, the launch procedure would additionally:

1. copy `full.template.yaml` to a protected runtime config outside Git;
2. inject the 11 private IDs from the gated manifest;
3. set and record the gated SHA and image digests;
4. run offline private fixture/isolation tests;
5. run metadata-only model catalogs;
6. produce a dry matrix proving exactly 440 unique identities;
7. require a final human review before invoking `run`.

No command that performs steps 1–7 has been run for the full suite.

## Runtime and spend projection

Observed Public-Pilot-4 completed-trial averages (public worlds only, so not guarantees for private worlds):

| Lane | Valid observations | Mean wall time/trial | Recorded solver spend | Linear 110-trial solver projection |
|---|---:|---:|---:|---:|
| GPT-5.6 Sol | 20 | 61.0 min | $106.12 | ~$583.65 |
| Claude Opus 5 | 20 | 52.9 min | $111.34 | ~$612.39 |
| Gemini 3.7 Flash | 20 | 33.3 min | $17.23 | ~$94.77 |
| Qwen3.5/DeepInfra | 9 | 9.7 min | $3.15 | ~$38.44, low confidence |

Approximate solver-only total: **~$1,329**, before judge calls, infrastructure-invalid partial attempts, provider price changes, or private-world difficulty. GPT/Claude input dominates cost because the full trajectory conversation grows each round.

With one active trial per provider lane, the observed means imply roughly 112 serial hours for the slowest lane. Allow **5–7 wall-clock days** for 440 trials plus fitting, transient provider failures, selective infrastructure retries, and aggregation. The Qwen estimate is unreliable because 11 of its 20 pilot identities remain infrastructure-invalid under DeepInfra shared-pool overload.

## Required decisions before launch

- authorize or reject the full 22-world/440-trial campaign;
- designate the person/account allowed to accept and hold gated artifacts;
- approve the private-world code/data separation described above;
- choose protected artifact storage/retention and sanitized-publication policy;
- decide the unresolved Qwen retry/provider condition without silently changing it;
- accept the approximate 5–7 day runtime and ~$1.3k+ solver/judge spend;
- decide whether any full result will be submitted to the official leaderboard (a separate external action).

## Cleanup

Use the explicit Compose project name already defined in `compose.yaml`:

```bash
docker compose down --remove-orphans
# Remove the named IPC volume only after all jobs and artifact checks complete:
docker volume rm discoverphysics-public-harness-ipc
```

Never delete or overwrite the protected run root as part of container cleanup.
