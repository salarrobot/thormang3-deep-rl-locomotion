from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from utils import load_runner, resolve_model, run_algo, run_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--map", action="store_true",
                    help="open a live telemetry/map window as well")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model_path, vec_path = resolve_model(args.run)
    cfg = run_config(args.run)
    model, venv, raw = load_runner(model_path, vec_path, cfg,
                                   run_algo(args.run), seed=args.seed)
    print(f"[live] {model_path} — close the viewer window to stop")

    dash = None
    if args.map:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from viz import Dashboard
        plt.ion()
        fig = plt.figure(figsize=(4.2, 8.4))
        fig.canvas.manager.set_window_title("THORMANG3 — live telemetry")
        dash = Dashboard(raw.dt, cfg.target_speed, fig=fig)
        fig.show()

    obs = venv.reset()
    R = np.zeros(9)
    t0 = time.perf_counter()
    with mujoco.viewer.launch_passive(raw.model, raw.data) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = raw._pelvis_id
            viewer.cam.distance, viewer.cam.elevation = 3.5, -15
        i = 0
        while viewer.is_running() and i * raw.dt < args.seconds:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = venv.step(act)
            info = infos[0]
            viewer.sync()

            if dash is not None:
                mujoco.mju_quat2Mat(R, raw.data.qpos[3:7])
                dash.update(i * raw.dt, raw.data.qpos[0:2], (R[0], R[3]),
                            info["v_forward"], raw._foot_contacts(),
                            info["push"], info["touchdowns"])
                if done[0]:
                    dash.trail.append((np.nan, np.nan))
                if i % 5 == 0:
                    dash.render()
                    dash.fig.canvas.flush_events()

            i += 1
            lag = t0 + i * raw.dt - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
    venv.close()
    print(f"[live] done — walked {raw.data.qpos[0]:.1f} m")


if __name__ == "__main__":
    main()
