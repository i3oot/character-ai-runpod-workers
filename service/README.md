# Kimodo motion and SkinTokens rigging on RunPod

Two separate queue workers following the existing Pixal3D setup. Models stay on
small, independent RunPod network volumes; input/output assets use the existing
private S3 artifact bucket. No model, mesh or credential is included in the image
source. The existing Pixal3D deployment is not changed.

## Current state — 28 October 2026

- Worker code and client implemented; seven CPU contract tests pass.
- Standard volumes created in EU-RO-1: Kimodo **25 GB** (`x70mhforid`),
  SkinTokens **10 GB** (`fwqrbw2hoo`, RunPod's minimum volume size).
- Public worker repository publication is authorized; container builds and live tests are in progress.
- Kimodo now uses the pinned public `matbee/kimodo-llm2vec-nf4` encoder.
  No gated Meta download or Hugging Face token is required.
- No live inference, model-cache download, VRAM measurement or export quality
  acceptance has been claimed.

The local deployment state lives in
`.tools/character-ai/runpod/deployment.json` and contains **only resource IDs and
non-secret settings**. Never copy private runtime environment files into a build.

## Cost and lifecycle

Planned GPU pool: `AMPERE_24` (24 GB L4/A5000/3090 class), one GPU per worker;
Blackwell MIG is excluded from this CUDA/Torch build. EU-RO-1 currently offers L4
in this price class. There is no automatic fallback to a higher priced pool.

Each endpoint: minimum zero workers, maximum one, five-second idle shutdown,
20-minute execution limit, serialized jobs. Published Flex pricing is about
**$0.69 per running worker-hour**; startup/model downloads and idle time are
billable. The cheaper 16 GB class leaves less headroom for SkinTokens and cannot
hold Kimodo's roughly 17 GB all-GPU setup. A measured 16 GB/CPU-offload trial can
be considered later without changing these defaults silently.

The separate 25+10 GB standard volumes cost approximately **$2.45/month** in total
at $0.07/GB/month, including periods with zero GPU workers. Inputs and results
are not written to those volumes. Existing object-storage and transfer charges
remain separate. Storage is retained until explicitly removed.

Sources: [GPU pricing](https://www.runpod.io/product/serverless),
[network volumes](https://docs.runpod.io/storage/network-volumes),
[RunPod API v2](https://docs.runpod.io/api-reference-v2/migrate-from-v1).

## Runtime credentials

The client reads the already provisioned private Pixal3D environment at
`.tools/image-to-3d/runpod/client.env` by default. `--env-file PATH` selects an
alternative. It parses assignments without executing a shell and never prints
their values. The worker receives S3 settings directly through its private
RunPod endpoint environment. RunPod API keys are client-only. Bucket permissions
must allow `character-ai/inputs/` and `character-ai/outputs/` in addition to the
existing Pixal3D prefixes.

Kimodo uses a pinned public Matbee NF4 export of Llama 3 plus LLM2Vec adapters.
The Llama 3 license still applies. The loader refuses to fall back to the gated
Meta model if the export is incomplete. No HF token is forwarded to the worker.
No Warden secret is retrieved or passed into a shell, file, environment or CLI.

## Client

Reuse the Pixal3D client virtual environment (boto3), or install its existing
`requirements-client.txt`. From the project root:

```sh
.tools/image-to-3d/runpod/client-venv/bin/python tools/character_ai/runpod_service/client.py \
  kimodo 'A person stands alert and looks around.' --duration 3 \
  -o art/animation-tests/nyx-alert

.tools/image-to-3d/runpod/client-venv/bin/python tools/character_ai/runpod_service/client.py \
  skintokens assets/characters/nyx/nyx.glb --use-skeleton \
  -o art/rig-tests/nyx-skin

.tools/image-to-3d/runpod/client-venv/bin/python tools/character_ai/runpod_service/client.py \
  fetch art/rig-tests/nyx-skin/job.json -o art/rig-tests/nyx-skin
```

Use an actual existing GLB path. Omitting `--use-skeleton` generates both skeleton
and weights; including it keeps an input skeleton and predicts skinning. The
SkinTokens adapter uses upstream texture/scale transfer and four weights per
vertex. It outputs `rigged.glb`. Kimodo outputs `motion.npz` and `motion.bvh`
(SOMA 77-joint skeleton, standard T-pose rest pose, BVH centimeters).
SkinTokens does not automatically produce a Kimodo-compatible skeleton, and this
service does not implement animation retargeting onto game rigs.

Use `--submit-only` to return after queuing. Keep `job.json` and reuse the same
command or `fetch`. The ticket is reserved **before** the billable API call.
An ambiguous response leaves a `submitting` ticket; inspect RunPod rather than
resubmitting. Local polling timeouts do not cancel the remote job. Fetch checks
durable S3 metadata before asking RunPod and validates output size/checksum.
An existing differing output is never overwritten silently.

Kimodo accepts a single prompt, 0.5–10 seconds, 10–200 diffusion steps, and
optional `--constraints constraints.json` in upstream constraint-list format.
SkinTokens accepts self-contained GLBs up to 128 MiB and 1–10 beams (default 5).
Failures/OOMs are not automatically resubmitted or moved to a pricier GPU.

## Build and deployment

`source-lock.json` pins the two upstream source revisions and primary model
checkpoints. Kimodo dependencies and Qwen config are
not a complete transitive lock. SkinTokens downloads only the two inference
checkpoints, not its training datasets or Qwen weight files.

The dedicated build repository contains `service/`, the unchanged Pixal3D
`storage.py` helper, and `.github/workflows/publish.yml`. Both Docker targets
publish model-free images. Never use the whole game workspace as build context.
Example in a staged build repository:

```sh
docker build -f service/Dockerfile --target kimodo -t kimodo-runpod-worker .
docker build -f service/Dockerfile --target skintokens -t skintokens-runpod-worker .
```

Run `deploy.py plan` for non-secret sizing. `deploy.py volumes` reconciles only
the two named caches. Once both immutable image tags have built and anonymous
pulling is verified, `deploy.py apply --image-tag COMMIT_SHA` creates the two
endpoints using the current RunPod **v2 management API**. It passes only the
necessary S3 runtime credentials to RunPod, never persists response environments,
and never resets an existing endpoint on rerun. A subsequent `deploy.py status`
prints resource metadata without credential values.

Before declaring the setup ready: run one short Kimodo generation and one small
SkinTokens rigging test; fetch and inspect real outputs; record peak VRAM and
GPU type; confirm both endpoints return to zero workers. Generated artifacts
still need Blender deformation/contact review and game export validation.

```sh
python -m unittest discover -s tools/character_ai/runpod_service -p 'test_*.py' -v
```
