from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np

from utils import load_runner, resolve_model, run_algo, run_config
from viz import hud


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--camera", default="track", choices=["track", "side"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    runs = [args.left, args.right]
    labels = args.labels or [run_algo(r).upper() for r in runs]
    out = args.out or args.left / "videos/race.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    sides = []
    for run in runs:
        model_path, vec_path = resolve_model(run)
        cfg = run_config(run)
        cfg.max_steps = int(args.seconds / 0.02) + 500
        model, venv, raw = load_runner(
            model_path, vec_path, cfg, run_algo(run),
            seed=args.seed, width=640, height=720, camera=args.camera)
        sides.append({"model": model, "venv": venv, "raw": raw,
                      "obs": venv.reset(), "falls": 0, "dist": 0.0})
        print(f"[race] {run} -> {model_path.name}")

    dt = sides[0]["raw"].dt
    n_steps = int(args.seconds / dt)
    writer = imageio.get_writer(str(out), fps=round(1 / dt),
                                codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=None)
    for i in range(n_steps):
        halves = []
        for s, label in zip(sides, labels):
            act, _ = s["model"].predict(s["obs"], deterministic=True)
            s["obs"], _, done, infos = s["venv"].step(act)
            info = infos[0]
            if done[0] and not info.get("TimeLimit.truncated", False):
                s["falls"] += 1
            live = float(np.linalg.norm(s["raw"].data.qpos[0:2]))
            s["dist"] = max(s["dist"], live)
            frame = s["raw"].render()
            frame = hud(frame,
                        [f"t     {i * dt:6.1f} s",
                         f"speed {info['v_forward']:+5.2f} m/s",
                         f"dist  {live:6.1f} m",
                         f"falls {s['falls']:3d}"],
                        push=info["push"], badge=label)
            halves.append(frame)
        writer.append_data(np.hstack(halves))
    writer.close()

    for s, label in zip(sides, labels):
        print(f"[race] {label}: dist {s['dist']:.1f} m, "
              f"falls {s['falls']}")
        s["venv"].close()
    print(f"[race] wrote {out}")


if __name__ == "__main__":
    main()
