# GEM-X video-to-motion on RunPod

Single-person video motion capture alongside the existing Kimodo and SkinTokens
queue workers. This uses NVIDIA GEM-X Body Motion native77, not a framewise pose
estimator or the separately gated EgoHand pipeline.

Preparation status (29 October 2026): local implementation and 13 CPU contract
tests pass. The container has not been built or deployed and no GPU smoke test
has run. Public repository push/CI needs explicit user approval after automatic
approval review rejected that publication step. Existing endpoints are unchanged.

## Operation

From the game workspace, with FFmpeg/ffprobe installed locally:

```sh
.tools/image-to-3d/runpod/client-venv/bin/python tools/character_ai/runpod_service/client.py \
  gemx /absolute/path/performance.mp4 --start 0 --duration 3 --camera static \
  -o art/animation-tests/video-performance

.tools/image-to-3d/runpod/client-venv/bin/python tools/character_ai/runpod_service/client.py \
  fetch art/animation-tests/video-performance/job.json -o art/animation-tests/video-performance
```

Use `--camera background` for moving cameras with enough visible background;
it does not silently fall back to a fixed camera if camera estimation fails.
For a tripod/fixed camera use `static`. The primary person is selected by total
detected screen area; this service does not select arbitrary people in a crowd.

Limits: MP4/MOV/WebM/MKV, 256 MiB upload, at most 4K source pixel count, selected
interval 0.5–10 seconds, 720-pixel longest side by default (adjustable 256–1280),
normalized 30 fps, no audio. An interval extending beyond the video is rejected.
Model input contains only the selected, resized interval. Inputs/results use the
existing private artifact bucket; the model volume holds only reusable caches.

`--submit-only`, ticket reservation before billing, ambiguous-submission refusal,
checksum-verified downloads and S3-first fetch work exactly as for the existing
workers. A timeout only stops local waiting. Resume the existing ticket, never
blindly submit a second job. No automatic OOM retry or more expensive GPU fallback.

## Outputs

- `motion.pt`: native GEM-X predictions, replayable observations, raw parameters
  and optional Contact/IK correction. Load only trusted PyTorch artifacts.
- `motion.npz`: numeric/unicode arrays, readable with `allow_pickle=False`:
  77 joint names/parents, world positions/rotations, fitted bind pose, SOMA
  parameters and 30 fps. Coordinates are Y-up, meters.
- `motion.bvh`: 77-joint skeleton without virtual Root, Y-up, centimeters,
  world-axis rest offsets. Six channels per joint preserve procedural joint
  translations rather than discarding them. Import into Blender with scale 0.01.
- `preview.mp4` and `preview.png`: source/2D observations beside a fixed front
  view of the reconstructed world skeleton, with the same frame count as BVH.
- `motion-report.json`: selection, provenance, coordinate conventions,
  correction diagnostics and limitations.

This produces motion, not new character meshes, rigging or weights. Retargeting
to our SkinTokens rigs, hand/body contact cleanup, loop creation and gameplay
root-motion handling remain separate Blender/export steps. Finger poses are
estimates from Body Motion; the dedicated Hand Pose model is not installed.

## Resources and cost

Dedicated `party-gemx-video-motion` endpoint, EU-RO-1, `AMPERE_16` only, one GPU,
minimum zero/maximum one worker, five-second idle shutdown, 20-minute job limit.
Separate 10 GB standard model-cache volume. Two ~3.5 GB image encoders run in
successive stages with batch size one; GPU suitability must be checked by a real
smoke run rather than inferred from model file size.

Published rates at setup: approximately $0.576 per running GPU hour, plus
$0.70/month for the retained cache. Cold starts/downloads and processing may be
billable; artifact storage/transfer is separate. No idle GPU is kept warm.
Sources: [RunPod endpoint pricing](https://docs.runpod.io/serverless/endpoints/endpoint-configurations),
[network volume pricing](https://docs.runpod.io/storage/network-volumes).

## Build and deployment

The dedicated public build repository contains only explicitly staged service
code and the existing storage helper, never game assets, weights or runtime
credentials. The workflow builds GEM-X on push; a manual dispatch can select
`kimodo` or `skintokens` without changing their live endpoints.

```sh
python tools/character_ai/runpod_service/deploy.py plan --service gemx
# After the immutable image has built and anonymous pull has been verified:
.tools/image-to-3d/runpod/client-venv/bin/python tools/character_ai/runpod_service/deploy.py \
  apply --service gemx --image-tag VERIFIED_40_CHARACTER_BUILD_COMMIT
```

GEM-X has a separate PyTorch 2.10/CUDA 12.8 image; the old workers retain their
runtimes. `source-lock.json` pins source, HF model revision, SOMA v0.3.3 asset
revision and SHA-256 checksums for the person detector. These files do not form
a complete transitive dependency lock. Public body weights need no HF token;
implicit HF token use is disabled in the GEM-X image. GEM-X code is Apache-2.0;
model use follows the NVIDIA Open Model License and component terms.

Before calling the worker tested: submit a short video, fetch all artifacts,
check finite motion arrays and frame counts, inspect the preview, and confirm
RunPod health reports no running workers afterward. CPU contract tests alone do
not establish motion quality or GPU fit.

```sh
python -m unittest discover -s tools/character_ai/runpod_service -p 'test_*.py' -v
```

CPU export tests require NumPy/SciPy; video normalization tests require FFmpeg.
Upstream: [GEM-X](https://github.com/NVlabs/GEM-X),
[model card](https://huggingface.co/nvidia/GEM-X).
