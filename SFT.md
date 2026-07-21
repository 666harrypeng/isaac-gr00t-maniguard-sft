# ManiGuard GR00T N1.6 SFT

This fork of Isaac-GR00T (`n1d6`) fine-tunes **GR00T N1.6** on the 6 ManiGuard base-task
families, packaged so you clone one repo, pull public datasets, and push public
checkpoints — no other codebase needed. It is the GR00T half of the ManiGuard VLA
benchmark (GR00T vs pi0.5 on identical data, cameras, and controller).

- **Model:** `nvidia/GR00T-N1.6-3B`, `NEW_EMBODIMENT`. Tuning = the N1.6 default:
  freeze the VLM (LLM + visual), train the projector + diffusion action head (no LoRA).
- **Inputs (benchmark parity with pi0.5):** 2 cameras (`image_left` overview + `wrist`),
  8-D joint state/action (7 arm + gripper), absolute JointController, NON_EEF, 16-step horizon.
- **Data:** the SAME public LeRobot datasets pi0.5 uses —
  `IDEAS-Lab-Northwestern/datagen-<fam>-v1-joint-5cam` (H.264 GOP10). GR00T only adds its
  own `meta/modality.json` + baked stats onto a lightweight prepared VIEW (symlinked
  videos/parquet); the shared dataset is never modified and no GR00T-specific dataset is
  published.

## Quick start

Two ways to get the env; pick one.

**Docker (for cluster submission):**
```bash
docker build -f docker/sft.Dockerfile -t gr00t-maniguard-sft .
docker run --gpus all --ipc=host -e HF_TOKEN -e WANDB_API_KEY \
  -v "$PWD/outputs:/workspace/outputs" gr00t-maniguard-sft --family jar
# or all six: ... gr00t-maniguard-sft --all
```

**Interactive (uv):**
```bash
uv sync --frozen --python 3.10          # x86_64: flash-attn/torchcodec/deepspeed as wheels
# (aarch64: run scripts/deployment/dgpu/install_deps.sh instead — builds torchcodec)
source .venv/bin/activate

# TRAIN (needs WANDB_API_KEY only); --data-root points at the shared LeRobot datasets
export WANDB_API_KEY=...
bash tools/gr00t_sft/run_all.sh --data-root /path/to/lerobot --family jar    # or --all

# PUSH later, separately (needs HF_TOKEN)
export HF_TOKEN=...
bash tools/gr00t_sft/run_all.sh --push --family jar                          # or --push --all
```

Per family, TRAIN = ensure the dataset is present (shared, read-only) → prepare a GR00T VIEW
(symlink videos/parquet + `modality.json` + baked stats) → train ~2 epochs. PUSH (`--push`)
uploads the latest checkpoint + card to HF. The shared dataset is never modified.

## Recipe (8-card config)

8 GPUs, DeepSpeed ZeRO-2, bf16, **global batch 256** (32/card), **2 epochs**,
`steps = ceil(frames*2/256)`, cosine LR with **peak 2e-4** (sqrt-scaled from GR00T's
`1e-4`), warmup 0.05, adamw_torch. `GLOBAL_BATCH_SIZE` must be divisible by the GPU count.

| family | frames | steps |
|---|---:|---:|
| clutter | 901,520 | 7,100 |
| jar | 946,870 | 7,400 |
| lid | 1,055,142 | 8,250 |
| dusty | 1,879,498 | 14,700 |
| stack | 2,652,083 | 20,750 |
| cabinet | 4,172,962 | 32,650 |

Tune per node without editing configs (`WORKERS` = **per-GPU** dataloader workers → ×GPUS total):
```bash
BATCH=256 GPUS=8 WORKERS=6 LR=2e-4 TAG=-run2 \
  bash tools/gr00t_sft/run_all.sh --data-root /path/to/lerobot --family jar
# or the per-run wrapper:
bash tools/gr00t_sft/run_sft.sh --dataset <prepped> --output <dir> --steps N --batch 256 --gpus 8 --lr 2e-4 --workers 6
```

## Outputs

`--push` uploads inference files + a model card to
`IDEAS-Lab-Northwestern/gr00t-n16-datagen-v1-<fam>-joint-2cam`. **wandb**: project
`maniguard-gr00tN1d6`, run `datagen_v1_<fam>_joint_2cam`. `TAG` appends to BOTH the HF repo
and the wandb project — e.g. `TAG=-run2` → HF `...-joint-2cam-run2`, wandb project
`maniguard-gr00tN1d6-run2` (run name unchanged, so families stay comparable within a project).

## Baking stats (maintainer)

Stats are shipped baked under `gr00t_stats/<fam>/`. To (re)bake from a local dataset copy
(CPU-only, minutes, no GPU/model):
```bash
python tools/gr00t_sft/bake_stats.py --data-root <root_with_datagen-*-v1-joint-5cam>
```
`prepare_dataset.py` uses the baked stats when present, else computes them on the node
(the training `DatasetFactory` also auto-generates missing stats, rank-0 + barrier).
