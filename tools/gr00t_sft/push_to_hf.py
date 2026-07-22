#!/usr/bin/env python
"""Push a GR00T-N1.6 SFT checkpoint + a generated model card to a public HF repo.

Points at the run dir whose ROOT holds the FINAL saved model; uploads only the
inference bundle (model safetensors + config.json + experiment_cfg/ + processor/)
and skips the intermediate checkpoint-*/ dirs, DeepSpeed ZeRO state (global_step*/,
*_states.pt), and optimizer / scheduler / rng / trainer-state files.
Generates a concise model card from the per-family metadata.

Run inside the Isaac-GR00T venv (needs HF_TOKEN). Usage:

    python tools/gr00t_sft/push_to_hf.py --ckpt <checkpoint_dir> \
        --repo IDEAS-Lab-Northwestern/gr00t-n16-datagen-v1-jar-joint-2cam \
        --title "Jar" --task jar \
        --data-repo IDEAS-Lab-Northwestern/datagen-jar-v1-joint-5cam \
        --frames 946870 --epochs 2 --steps 7400 --batch 256
"""

import argparse

from huggingface_hub import HfApi

# Training-only artifacts — not needed for inference; skip to keep the repo lean. When --ckpt
# is a run dir, "checkpoint-*" drops every intermediate step checkpoint (each re-saves the full
# weights + a ~30GB DeepSpeed global_step*/); the DeepSpeed + resume-state patterns below also
# strip those if --ckpt points straight at a single checkpoint-<step>/ dir.
_IGNORE = [
    "checkpoint-*",       # intermediate HF-Trainer step checkpoints (keep only the final root model)
    "global_step*",       # DeepSpeed consolidated state dirs
    "*optim_states.pt",   # DeepSpeed ZeRO optimizer shards
    "*model_states.pt",   # DeepSpeed model-state shards
    "zero_to_fp32.py",
    "latest",
    "optimizer.pt",
    "rng_state*.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "wandb_config.json",
]

_CARD = """---
license: apache-2.0
base_model: nvidia/GR00T-N1.6-3B
pipeline_tag: robotics
tags: [robotics, vla, gr00t, gr00t-n1.6, manipulation, maniguard, franka]
---

# GR00T-N1.6 - {title} (joint, 2-cam)

NVIDIA Isaac **GR00T-N1.6-3B** fine-tuned on the ManiGuard **{task}** base task (sim
Franka Panda). Part of the ManiGuard VLA benchmark - GR00T vs pi0.5 on the same task
families with identical data, cameras, and controller.

## Model
- **Base:** [nvidia/GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B) - Eagle (nvidia/Eagle-Block2A-2B-v2) VLM + flow-matching DiT action head
- **Embodiment:** NEW_EMBODIMENT - Franka Panda, **8-D joint** state/action (7 arm joints + 1 gripper)
- **Cameras (2):** image_left (overview) + wrist (256x256)
- **Action:** arm = state-relative chunks, gripper = absolute; 16-step horizon; NON_EEF (joint space)
- **Tuning:** GR00T-N1.6 default - VLM (LLM + visual) **frozen**, train projector + diffusion action head (**no LoRA**)

## Training
- 8-card config, DeepSpeed ZeRO-2, bf16, global batch {batch}, {steps} steps (~{epochs} epochs over {frames:,} frames), cosine LR (peak 2e-4, sqrt-scaled), warmup 0.05
- Data: [{data_repo}](https://huggingface.co/datasets/{data_repo}); videos decoded as H.264 for GR00T's torchcodec loader

## Usage
Load with `Gr00tPolicy` from [Isaac-GR00T (n1d6)](https://github.com/NVIDIA/Isaac-GR00T/tree/n1d6), `--embodiment-tag NEW_EMBODIMENT`. The included `experiment_cfg/` carries the modality config + normalization stats.

> WARNING - Convention (must match at eval): joint-space JointController (absolute joint targets, NON_EEF) + 2 cameras (image_left overview + wrist). A mismatched controller or camera set silently feeds an out-of-distribution input.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir to upload")
    ap.add_argument("--repo", required=True, help="target HF model repo (org/name)")
    ap.add_argument("--title", required=True, help='card title, e.g. "Stack-Retrieve"')
    ap.add_argument("--task", required=True, help="task slug, e.g. stack-retrieve")
    ap.add_argument("--data-repo", required=True, help="source HF dataset repo")
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=args.ckpt,
        repo_id=args.repo,
        repo_type="model",
        ignore_patterns=_IGNORE,
        commit_message=f"GR00T-N1.6 SFT on {args.task} ({args.epochs} epochs, {args.steps} steps)",
    )
    card = _CARD.format(
        title=args.title,
        task=args.task,
        data_repo=args.data_repo,
        frames=args.frames,
        epochs=args.epochs,
        steps=args.steps,
        batch=args.batch,
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="model card",
    )
    print("PUSHED", args.repo)


if __name__ == "__main__":
    main()
