# CLAUDE.md — GR00T N1.6 SFT fork

Guidance for an agent running SFT on a compute node. Full runbook: `SFT.md`.

## What this is
Isaac-GR00T (`n1d6`) + a vendored ManiGuard SFT layer (`maniguard/gr00t_sft/`,
`tools/gr00t_sft/`, `gr00t_stats/`). It SFTs GR00T N1.6 on 6 families with inputs identical
to the pi0.5 track (2-cam, 8-D joint). Do NOT change the cameras, controller, or action
representation — parity is the point.

## Before a full run — verify on THIS node
- Env: `python -c "import flash_attn, deepspeed, torchcodec; print('env ok')"` and
  `ffmpeg -version | head -1`. (x86_64 gets these as wheels via `uv sync`; aarch64 uses
  `scripts/deployment/dgpu/install_deps.sh`.)
- Embodiment registers: `python -c "import sys; sys.path.insert(0,'.'); import maniguard.gr00t_sft.maniguard_embodiment; from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS; print('new_embodiment' in MODALITY_CONFIGS)"` → `True`.
- `GLOBAL_BATCH_SIZE` divisible by GPU count (`run_sft.sh` asserts). Default 256 / 8 = 32/card.
- Tokens set: `HF_TOKEN` (dataset pull + push), `WANDB_API_KEY` (logs).

## Recipe (do not silently change)
8-GPU DeepSpeed ZeRO-2, global batch 256, 2 epochs, cosine LR **peak 2e-4** (sqrt-scaled
from GR00T's 1e-4), warmup 0.05, bf16. Freeze = upstream default (projector + diffusion
head; LLM + visual frozen; no LoRA) — pass NO tuning flags.

## Quality gate (first family)
On the first (small) family, watch train loss + grad_norm for the first few hundred steps.
If unstable (divergence / grad_norm spikes), fall back to peak LR 1e-4 and record it:
`bash tools/gr00t_sft/run_sft.sh ... --lr 1e-4`. Never trade quality for speed.

## Rules
- **Never disclose hardware** (GPU model/count as a spec) in configs, commits, or model cards.
  "8-card config" phrasing only.
- Don't mutate the shared datasets; `prepare_dataset.py` symlinks videos/parquet into a VIEW.
- Push to `gr00t-n16-datagen-v1-<fam>-joint-2cam` (+ `TAG=` to distinguish runs).
- Run: `bash tools/gr00t_sft/run_all.sh --family <fam>` (or `--all`); one family = swap the name.
