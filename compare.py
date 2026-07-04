from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_plots import (AQUA, BASELINE, BLUE, GRID, INK, INK_2,
                        MUTED, SURFACE, fmt_steps, load_monitor,
                        load_progress, rolling, save)
from utils import load_model, resolve_model

COLORS = [BLUE, AQUA]


def load_eval(run: Path) -> dict | None:
    f = run / "eval/evaluations.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {"t": d["timesteps"], "r": d["results"],
            "len": d["ep_lengths"]}


def steps_to_hours(run: Path) -> tuple[np.ndarray, np.ndarray]:
    prog = load_progress(run)
    d = prog[["time/total_timesteps", "time/time_elapsed"]].dropna()
    return (d["time/total_timesteps"].to_numpy(),
            d["time/time_elapsed"].to_numpy() / 3600)


def _end_label(ax, x, y, text, color) -> None:
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, 0),
                color=INK_2, fontsize=9, fontweight="bold",
                va="center", annotation_clip=False)
    ax.plot([x], [y], "o", ms=4, color=color)


def plot_eval_reward(runs, labels, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    zoom = min((load_eval(r)["t"][-1] for r in runs
                if load_eval(r) is not None), default=4e6) * 1.05
    for ax, xmax, title in ((axes[0], None, "full training"),
                            (axes[1], zoom, "early training (zoom)")):
        for run, label, color in zip(runs, labels, COLORS):
            ev = load_eval(run)
            if ev is None:
                continue
            m, s = ev["r"].mean(1), ev["r"].std(1)
            ax.fill_between(ev["t"], m - s, m + s, color=color, alpha=0.15,
                            linewidth=0)
            ax.plot(ev["t"], m, color=color, label=label)
            k = (np.searchsorted(ev["t"], xmax) - 1 if xmax
                 else len(ev["t"]) - 1)
            k = max(0, min(k, len(ev["t"]) - 1))
            _end_label(ax, ev["t"][k], m[k], label, color)
        ax.set_title(f"Evaluation reward — {title}")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("episode reward")
        ax.xaxis.set_major_formatter(fmt_steps)
        ax.set_xlim(0, xmax)
        ax.legend(loc="lower right")
    save(fig, out, "cmp_01_eval_reward.png")


def plot_wallclock(runs, labels, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for run, label, color in zip(runs, labels, COLORS):
        ev = load_eval(run)
        if ev is None:
            continue
        steps, hours = steps_to_hours(run)
        h = np.interp(ev["t"], steps, hours)
        m, s = ev["r"].mean(1), ev["r"].std(1)
        ax.fill_between(h, m - s, m + s, color=color, alpha=0.15,
                        linewidth=0)
        ax.plot(h, m, color=color, label=label)
        _end_label(ax, h[-1], m[-1], label, color)
    ax.set_title("Evaluation reward vs wall-clock time (same machine)")
    ax.set_xlabel("training time (hours)")
    ax.set_ylabel("episode reward")
    ax.legend(loc="lower right")
    save(fig, out, "cmp_02_wallclock.png")


def plot_training(runs, labels, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for run, label, color in zip(runs, labels, COLORS):
        mon = load_monitor(run)
        if mon is None:
            continue
        axes[0].plot(mon["steps"], rolling(mon["ep_speed"].to_numpy()),
                     color=color, label=label)
        axes[1].plot(mon["steps"], rolling(mon["l"].to_numpy()),
                     color=color, label=label)
    axes[0].set_title("Walking speed over training (rolling mean)")
    axes[0].set_ylabel("episode average speed (m/s)")
    axes[1].set_title("Episode length over training (rolling mean)")
    axes[1].set_ylabel("steps per episode")
    for ax in axes:
        ax.set_xlabel("environment steps")
        ax.xaxis.set_major_formatter(fmt_steps)
        ax.legend(loc="lower right")
    save(fig, out, "cmp_03_training.png")


def rollout_speed(run: Path) -> str:
    npz = run / "videos/demo.npz"
    if not npz.exists():
        return "n/a"
    d = np.load(npz)
    return f"{float(np.mean(d['v_fwd'])):.2f} m/s"


def summary_rows(runs, labels) -> pd.DataFrame:
    rows = []
    for run, label in zip(runs, labels):
        ev = load_eval(run)
        steps, hours = steps_to_hours(run)
        model_path, _ = resolve_model(run)
        model = load_model(model_path, label.lower()
                           if label.lower() in ("ppo", "sac") else None)
        n_params = sum(p.numel() for p in model.policy.parameters())
        m = ev["r"].mean(1)
        rows.append({
            "algorithm": label,
            "env steps": f"{steps.max() / 1e6:.1f} M",
            "wall time": f"{hours.max():.2f} h",
            "best eval reward": f"{m.max():,.0f}",
            "final eval reward": f"{m[-1]:,.0f}",
            "final eval ep len": f"{ev['len'][-1].mean():,.0f} / 1000",
            "walk speed (demo)": rollout_speed(run),
            "policy params": f"{n_params / 1e3:,.0f} k",
        })
    return pd.DataFrame(rows)


def plot_table(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 0.62 * (len(df) + 1) + 0.5))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.7)
    for (row, _), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(color=INK, fontweight="bold")
            cell.set_facecolor("#f0efec")
        else:
            cell.set_text_props(color=INK_2)
            cell.set_facecolor(SURFACE)
    ax.set_title("Algorithm comparison — THORMANG3 walking",
                 color=INK, fontsize=12, pad=16)
    save(fig, out, "cmp_04_table.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs/comparison"))
    args = ap.parse_args()
    labels = args.labels or [r.name.upper() for r in args.runs]
    assert len(labels) == len(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[compare] {', '.join(labels)} -> {args.out}/")

    plot_eval_reward(args.runs, labels, args.out)
    plot_wallclock(args.runs, labels, args.out)
    plot_training(args.runs, labels, args.out)
    df = summary_rows(args.runs, labels)
    plot_table(df, args.out)
    (args.out / "comparison.md").write_text(
        "# Algorithm comparison\n\n" + df.to_markdown(index=False) + "\n")
    print(df.to_string(index=False))
    print("[compare] done")


if __name__ == "__main__":
    main()
