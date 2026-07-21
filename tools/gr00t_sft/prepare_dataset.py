#!/usr/bin/env python
"""Make a GR00T-ready view of a ManiGuard LeRobot dataset WITHOUT copying big files
or mutating the source: symlink videos/ + data/, materialize a real meta/ (LeRobot
meta + GR00T's modality.json + stats), idempotent. Videos are already H.264 (GOP10)
so there is NO transcode. Run in the Isaac-GR00T venv.

    python tools/gr00t_sft/prepare_dataset.py --src <lerobot_dir> --out <prepped_dir> \
        [--stats-dir <fork>/gr00t_stats/<fam>]

Stats resolution: if <stats-dir>/stats.json exists, copy it (+ relative_stats.json)
into meta/ (baked, no compute); else compute via gr00t.data.stats (the embodiment must
be registered first -- this module imports it, which registers NEW_EMBODIMENT).
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

# tools/gr00t_sft/prepare_dataset.py -> repo root -> the embodiment config file.
_REPO = Path(__file__).resolve().parents[2]
_CFG = _REPO / "maniguard" / "gr00t_sft" / "maniguard_embodiment.py"


def _load_embodiment():
    """Exec the self-contained config file: registers NEW_EMBODIMENT, returns module."""
    spec = importlib.util.spec_from_file_location("maniguard_embodiment", _CFG)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load embodiment config from {_CFG}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _link(src: Path, dst: Path) -> None:
    """Symlink dst -> src (idempotent). No-op if dst already exists."""
    if dst.is_symlink() or dst.exists():
        return
    dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def prepare(src: Path, out: Path, stats_dir: Path | None, mod) -> None:
    if not (src / "meta").is_dir():
        raise FileNotFoundError(f"{src}/meta not found -- not a LeRobot v2 dataset")
    out.mkdir(parents=True, exist_ok=True)

    # 1. symlink the heavy, content-bearing dirs (no copy; the source stays untouched).
    _link(src / "videos", out / "videos")
    _link(src / "data", out / "data")

    # 2. real meta/ = copy LeRobot's meta, then add GR00T's files (isolated from src/meta).
    (out / "meta").mkdir(exist_ok=True)
    for f in (src / "meta").glob("*"):
        d = out / "meta" / f.name
        if f.is_file() and not d.exists():
            shutil.copy2(f, d)
    (out / "meta" / "modality.json").write_text(json.dumps(mod.MODALITY_JSON, indent=4) + "\n")

    # 3. stats: baked-copy if available, else compute (embodiment already registered).
    baked = stats_dir is not None and (stats_dir / "stats.json").is_file()
    if baked:
        shutil.copy2(stats_dir / "stats.json", out / "meta" / "stats.json")
        rel = stats_dir / "relative_stats.json"
        if rel.is_file():
            shutil.copy2(rel, out / "meta" / "relative_stats.json")
        print(f"[prepare] used baked stats from {stats_dir}")
    elif not (out / "meta" / "stats.json").is_file():
        from gr00t.data.stats import main as stats_main
        from gr00t.data.types import EmbodimentTag

        print(f"[prepare] no baked stats -- computing via gr00t.data.stats for {out}")
        stats_main(str(out), EmbodimentTag.NEW_EMBODIMENT)  # CPU-only; writes meta/stats + relative_stats
    print(f"[prepare] done: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="source LeRobot dataset dir (untouched)")
    ap.add_argument("--out", required=True, help="output GR00T-ready view dir (symlinks + real meta)")
    ap.add_argument("--stats-dir", default=None, help="fork gr00t_stats/<fam> with baked stats")
    args = ap.parse_args()

    mod = _load_embodiment()  # register NEW_EMBODIMENT once
    prepare(Path(args.src), Path(args.out), Path(args.stats_dir) if args.stats_dir else None, mod)


if __name__ == "__main__":
    main()
