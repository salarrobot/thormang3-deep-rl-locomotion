from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
RED = "#e34948"
BLUE_LIGHT = "#9ec5f4"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 160,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0,
})

fmt_steps = FuncFormatter(
    lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k" if x else "0")


def rolling(series: np.ndarray, frac: float = 0.02,
            min_w: int = 5) -> np.ndarray:
    w = max(min_w, int(len(series) * frac)) | 1
    return pd.Series(series).rolling(w, min_periods=1, center=True).mean()


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name)
    plt.close(fig)
    print(f"  {name}")


def load_progress(run: Path) -> pd.DataFrame | None:
    f = run / "logs/progress.csv"
    return pd.read_csv(f) if f.exists() else None


def load_monitor(run: Path) -> pd.DataFrame | None:
    frames = []
    for f in sorted((run / "monitor").glob("*.monitor.csv")):
        try:
            frames.append(pd.read_csv(f, skiprows=1))
        except Exception:
            pass
    if not frames:
        return None
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()
    return df


def plot_training_curves(prog: pd.DataFrame, out: Path) -> None:
    for col, name, title, ylabel in (
            ("rollout/ep_rew_mean", "01_reward.png",
             "Training: mean episode reward", "episode reward"),
            ("rollout/ep_len_mean", "02_episode_length.png",
             "Training: mean episode length", "steps per episode (50 Hz)")):
        d = prog[["time/total_timesteps", col]].dropna()
        fig, ax = plt.subplots(figsize=(8, 4.2))
        x, y = d["time/total_timesteps"], d[col]
        ax.plot(x, y, color=BLUE_LIGHT, lw=1.2)
        ax.plot(x, rolling(y.to_numpy()), color=BLUE)
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(fmt_steps)
        ax.set_xlim(left=0)
        save(fig, out, name)


def plot_eval(run: Path, out: Path) -> None:
    f = run / "eval/evaluations.npz"
    if not f.exists():
        return
    d = np.load(f)
    t, res = d["timesteps"], d["results"]
    mean, std = res.mean(1), res.std(1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(t, mean - std, mean + std, color=BLUE, alpha=0.15,
                    linewidth=0)
    ax.plot(t, mean, color=BLUE, marker="o", ms=3.5)
    ax.set_title("Deterministic evaluation reward (mean ± std of 5 episodes)")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward")
    ax.xaxis.set_major_formatter(fmt_steps)
    ax.set_xlim(left=0)
    save(fig, out, "03_eval_reward.png")


def plot_ppo_diagnostics(prog: pd.DataFrame, out: Path) -> None:
    panels = [("train/approx_kl", "approx. KL"),
              ("train/clip_fraction", "clip fraction"),
              ("train/entropy_loss", "entropy loss"),
              ("train/explained_variance", "explained variance"),
              ("train/value_loss", "value loss"),
              ("train/policy_gradient_loss", "policy gradient loss"),
              ("train/std", "action std"),
              ("train/learning_rate", "learning rate")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    for ax, (col, title) in zip(axes.flat, panels):
        if col not in prog:
            ax.axis("off")
            continue
        d = prog[["time/total_timesteps", col]].dropna()
        ax.plot(d["time/total_timesteps"], d[col], color=BLUE, lw=1.4)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(fmt_steps)
        ax.tick_params(labelsize=8)
    fig.suptitle("PPO optimizer diagnostics", fontsize=12)
    save(fig, out, "04_ppo_diagnostics.png")


def _grid(n: int, ncols: int = 4) -> tuple[plt.Figure, np.ndarray]:
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.2 * ncols, 2.3 * nrows + 0.6),
                             sharex=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes


def plot_reward_components_training(mon: pd.DataFrame, out: Path) -> None:
    comps = [c for c in mon.columns if c.startswith("rc_")]
    if not comps:
        return
    fig, axes = _grid(len(comps))
    for ax, c in zip(axes.flat, comps):
        per_step = mon[c] / mon["l"]
        ax.plot(mon["steps"], per_step, color=BLUE_LIGHT, lw=0.6)
        ax.plot(mon["steps"], rolling(per_step.to_numpy()), color=BLUE)
        ax.set_title(c.replace("rc_", "reward: "), fontsize=10)
        ax.xaxis.set_major_formatter(fmt_steps)
        ax.tick_params(labelsize=8, labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps")
    fig.suptitle("Reward components over training (per-step average)",
                 fontsize=12)
    save(fig, out, "05_reward_components.png")


def plot_speed_distance(mon: pd.DataFrame, out: Path) -> None:
    if "ep_speed" not in mon.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title, ylab in (
            (axes[0], "ep_speed", "Episode average forward speed",
             "speed (m/s)"),
            (axes[1], "ep_dist", "Episode forward distance",
             "distance (m)")):
        ax.plot(mon["steps"], mon[col], color=BLUE_LIGHT, lw=0.6)
        ax.plot(mon["steps"], rolling(mon[col].to_numpy()), color=BLUE)
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylab)
        ax.xaxis.set_major_formatter(fmt_steps)
    save(fig, out, "06_speed_distance.png")


def longest_segment(done: np.ndarray) -> slice:
    bounds = [0, *(np.flatnonzero(done) + 1), len(done)]
    seg = max(zip(bounds[:-1], bounds[1:]), key=lambda ab: ab[1] - ab[0])
    return slice(*seg)


def plot_velocity(roll: dict, target: float, out: Path) -> None:
    t, v = roll["t"], roll["v_fwd"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, v, color=BLUE_LIGHT, lw=0.8)
    ax.plot(t, rolling(v, 0.01, 11), color=BLUE)
    ax.axhline(target, color=MUTED, ls="--", lw=1.2)
    ax.text(t[-1], target, f"  target {target:g} m/s", color=INK_2,
            va="bottom", ha="right", fontsize=9)
    for i in np.flatnonzero(roll["done"]):
        ax.axvline(t[i], color=BASELINE, lw=0.8)
    ax.set_title("Rollout: forward speed")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    save(fig, out, "07_rollout_speed.png")


def plot_height_upright(roll: dict, out: Path) -> None:
    t = roll["t"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t, roll["height"], color=BLUE)
    axes[0].set_title("Pelvis height")
    axes[0].set_ylabel("z (m)")
    axes[1].plot(t, roll["upright"], color=BLUE)
    axes[1].set_title("Uprightness (cos of torso tilt)")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("time (s)")
        for i in np.flatnonzero(roll["done"]):
            ax.axvline(t[i], color=BASELINE, lw=0.8)
    save(fig, out, "08_height_upright.png")


def plot_gait_diagram(roll: dict, out: Path) -> None:
    seg = longest_segment(roll["done"])
    t, c = roll["t"][seg], roll["contact"][seg]
    if len(t) > 500:
        t, c = t[:500], c[:500]
    dt = float(roll["dt"])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for row, (label, color) in enumerate((("left foot", BLUE),
                                          ("right foot", AQUA))):
        on = c[:, row].astype(bool)
        edges = np.flatnonzero(np.diff(np.r_[0, on, 0].astype(int)))
        for start, stop in zip(edges[::2], edges[1::2]):
            ax.barh(1 - row, (stop - start) * dt, left=t[0] + start * dt,
                    height=0.55, color=color, edgecolor=SURFACE, lw=0.5)
    ax.set_yticks([1, 0], ["left foot", "right foot"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("Gait diagram — filled = stance, empty = swing")
    ax.set_xlabel("time (s)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "09_gait_diagram.png")


def plot_joint_angles(roll: dict, out: Path) -> None:
    seg = longest_segment(roll["done"])
    start = seg.start + min(100, max(0, (seg.stop - seg.start) - 200))
    window = slice(start, min(start + 200, seg.stop))
    t = roll["t"][window]
    q = np.degrees(roll["q"][window])
    names = [("hip_y", 0), ("hip_r", 1), ("hip_p", 2),
             ("kn_p", 3), ("an_p", 4), ("an_r", 5)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, (jname, k) in zip(axes.flat, names):
        ax.plot(t, q[:, k], color=BLUE, label="left")
        ax.plot(t, q[:, 6 + k], color=AQUA, label="right")
        ax.set_title(jname, fontsize=10)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("angle (deg)")
    fig.suptitle("Joint angles during walking (blue = left, green = right)",
                 fontsize=12)
    save(fig, out, "10_joint_angles.png")


def plot_torques(roll: dict, out: Path) -> None:
    tau = roll["tau"]
    names = [str(n) for n in roll["joint_names"]]
    rms = np.sqrt((tau ** 2).mean(0))
    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(names))[::-1]
    colors = [BLUE if n.startswith("l_") else AQUA for n in names]
    ax.barh(ypos, rms, height=0.62, color=colors)
    for y, v in zip(ypos, rms):
        ax.text(v + 0.4, y, f"{v:.1f}", va="center", color=INK_2, fontsize=8.5)
    ax.set_yticks(ypos, [n.replace("_leg", "") for n in names], fontsize=9)
    ax.set_title("RMS actuator torque per joint (blue = left, green = right)")
    ax.set_xlabel("torque (N·m)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "11_torque_rms.png")


def plot_action_heatmap(roll: dict, out: Path) -> None:
    seg = longest_segment(roll["done"])
    a = roll["action"][seg][:500]
    t0 = roll["t"][seg][0]
    names = [str(n).replace("_leg", "") for n in roll["joint_names"]]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    im = ax.imshow(a.T, aspect="auto", cmap=DIVERGING, vmin=-1, vmax=1,
                   extent=(t0, t0 + len(a) * float(roll["dt"]),
                           len(names) - 0.5, -0.5), interpolation="nearest")
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.set_title("Policy actions (offset from standing pose, normalized)")
    ax.set_xlabel("time (s)")
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.outline.set_visible(False)
    save(fig, out, "12_action_heatmap.png")


def plot_trajectory(roll: dict, out: Path) -> None:
    xy = roll["base_xy"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axhline(0, color=MUTED, ls="--", lw=1)
    ax.plot(xy[:, 0], xy[:, 1], color=BLUE)
    ax.plot(xy[0, 0], xy[0, 1], "o", color=INK, ms=6)
    ax.annotate("start", (xy[0, 0], xy[0, 1]), textcoords="offset points",
                xytext=(6, 6), color=INK_2, fontsize=9)
    ax.plot(xy[-1, 0], xy[-1, 1], "s", color=VIOLET, ms=6)
    ax.annotate("end", (xy[-1, 0], xy[-1, 1]), textcoords="offset points",
                xytext=(6, 6), color=INK_2, fontsize=9)
    ax.set_title("Top-down base trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    save(fig, out, "13_trajectory.png")


def plot_rollout_components(roll: dict, out: Path) -> None:
    comp, names = roll["comp"], [str(n) for n in roll["comp_names"]]
    t = roll["t"]
    fig, axes = _grid(len(names))
    for ax, k in zip(axes.flat, range(len(names))):
        ax.plot(t, comp[:, k], color=BLUE, lw=1.2)
        ax.set_title(names[k].replace("rc_", "reward: "), fontsize=10)
        ax.tick_params(labelsize=8, labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Reward components during the rollout", fontsize=12)
    save(fig, out, "14_rollout_components.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--rollout", type=Path, default=None,
                    help="rollout .npz (default <run>/videos/demo.npz)")
    args = ap.parse_args()
    out = args.run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[plots] writing to {out}/")

    prog = load_progress(args.run)
    if prog is not None:
        plot_training_curves(prog, out)
        plot_ppo_diagnostics(prog, out)
    plot_eval(args.run, out)

    mon = load_monitor(args.run)
    if mon is not None and len(mon) > 10:
        plot_reward_components_training(mon, out)
        plot_speed_distance(mon, out)

    npz = args.rollout or args.run / "videos/demo.npz"
    if npz.exists():
        roll = dict(np.load(npz))
        cfg_file = args.run / "config.json"
        target = (json.loads(cfg_file.read_text())["cfg"]["target_speed"]
                  if cfg_file.exists() else 0.8)
        plot_velocity(roll, target, out)
        plot_height_upright(roll, out)
        plot_gait_diagram(roll, out)
        plot_joint_angles(roll, out)
        plot_torques(roll, out)
        plot_action_heatmap(roll, out)
        plot_trajectory(roll, out)
        plot_rollout_components(roll, out)
    else:
        print(f"[plots] no rollout data at {npz} — run record_video.py first")
    print("[plots] done")


if __name__ == "__main__":
    main()
