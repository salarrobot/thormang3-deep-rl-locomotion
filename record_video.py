from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore", message="Ignoring fixed")

import imageio.v2 as imageio
import mujoco
import numpy as np

from env import LEG_JOINTS
from utils import load_runner, resolve_model, run_algo, run_config
from viz import Dashboard, compose, hud

PANEL_W = 320


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, help="run directory")
    ap.add_argument("--model", type=Path, help="explicit model .zip")
    ap.add_argument("--vecnorm", type=Path, help="explicit vecnormalize .pkl")
    ap.add_argument("--out", type=Path, help="output .mp4 path")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--camera", default="track", choices=["track", "side"])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--dashboard", action=argparse.BooleanOptionalAction,
                    default=True, help="live map/telemetry side panel")
    ap.add_argument("--badge", default=None,
                    help="top-right label (default: algorithm name)")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.model and args.vecnorm:
        model_path, vec_path = args.model, args.vecnorm
        run_dir = args.model.parent.parent
    else:
        assert args.run, "pass --run, or --model with --vecnorm"
        run_dir = args.run
        model_path, vec_path = resolve_model(run_dir)
    out_dir = run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.out or out_dir / "demo.mp4"
    npz_path = video_path.with_suffix(".npz")

    cfg = run_config(run_dir)
    cfg.max_steps = int(args.seconds / 0.02) + 500
    algo = run_algo(run_dir)
    badge = args.badge or algo.upper()
    w3d = args.width - PANEL_W if args.dashboard else args.width
    model, venv, raw = load_runner(model_path, vec_path, cfg, algo,
                                   seed=args.seed, width=w3d,
                                   height=args.height, camera=args.camera)
    print(f"[video] model={model_path}  algo={algo}  camera={args.camera}")

    dash = (Dashboard(raw.dt, cfg.target_speed, PANEL_W, args.height)
            if args.dashboard else None)
    n_steps = int(args.seconds / raw.dt)
    log: dict[str, list] = {k: [] for k in
                            ("t", "q", "dq", "tau", "action", "target",
                             "reward", "comp", "v_fwd", "height", "upright",
                             "contact", "base_xy", "foot_z", "push", "done")}
    obs = venv.reset()
    writer = imageio.get_writer(str(video_path), fps=round(1 / raw.dt),
                                codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=None)
    R = np.zeros(9)
    for i in range(n_steps):
        act, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, rew, done, infos = venv.step(act)
        info = infos[0]
        t = i * raw.dt
        d = raw.data
        contacts = raw._foot_contacts()

        frame = raw.render()
        frame = hud(frame,
                    [f"t     {t:6.1f} s",
                     f"speed {info['v_forward']:+5.2f} m/s",
                     f"dist  {np.linalg.norm(d.qpos[0:2]):6.1f} m"],
                    push=info["push"], badge=badge)
        if dash is not None:
            mujoco.mju_quat2Mat(R, d.qpos[3:7])
            dash.update(t, d.qpos[0:2], (R[0], R[3]), info["v_forward"],
                        contacts, info["push"], info["touchdowns"])
            if done[0]:
                dash.trail.append((np.nan, np.nan))
            frame = compose(frame, dash.render())
        writer.append_data(frame)

        log["t"].append(t)
        log["q"].append(d.qpos[raw._qadr].copy())
        log["dq"].append(d.qvel[raw._vadr].copy())
        log["tau"].append(d.actuator_force[raw._act_ids].copy())
        log["action"].append(np.array(act[0], np.float32))
        log["target"].append(d.ctrl[raw._act_ids].copy())
        log["reward"].append(float(rew[0]))
        log["comp"].append([info["reward_components"][k]
                            for k in sorted(info["reward_components"])])
        log["v_fwd"].append(info["v_forward"])
        log["height"].append(info["height"])
        log["upright"].append(info["upright"])
        log["contact"].append(contacts)
        log["base_xy"].append(d.qpos[0:2].copy())
        log["foot_z"].append([d.geom_xpos[g][2] for g in raw._foot_geoms])
        log["push"].append(info["push"] or (np.nan, np.nan))
        log["done"].append(bool(done[0]))
    writer.close()

    arrays = {k: np.asarray(v, dtype=float) if k != "done"
              else np.asarray(v) for k, v in log.items()}
    arrays["comp_names"] = np.array(
        sorted(infos[0]["reward_components"].keys()))
    arrays["joint_names"] = np.array(LEG_JOINTS)
    arrays["dt"] = np.float64(raw.dt)
    np.savez_compressed(npz_path, **arrays)

    n_falls = int(arrays["done"].sum())
    n_pushes = int(np.isfinite(arrays["push"][:, 0]).sum())
    print(f"[video] wrote {video_path} ({n_steps} frames, "
          f"{n_steps * raw.dt:.0f} s), falls={n_falls}, "
          f"push-frames={n_pushes}")
    print(f"[video] rollout data -> {npz_path}  "
          f"(mean fwd speed {np.mean(arrays['v_fwd']):.2f} m/s)")
    venv.close()


if __name__ == "__main__":
    main()
