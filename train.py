#!/usr/bin/env python



from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize)

from env import RC_KEYS, ThormangWalkEnv, WalkConfig

ALGOS = {"ppo": PPO, "sac": SAC}


def make_env(cfg: WalkConfig, monitor_dir: Path | None, rank: int, seed: int):
    def _init():
        env = ThormangWalkEnv(cfg=cfg)
        env.reset(seed=seed + rank)
        fname = str(monitor_dir / f"env_{rank:02d}") if monitor_dir else None
        return Monitor(env, filename=fname, info_keywords=RC_KEYS)
    return _init


def build_vec_env(cfg: WalkConfig, n_envs: int, seed: int,
                  monitor_dir: Path | None, training: bool,
                  norm_reward: bool) -> VecNormalize:
    fns = [make_env(cfg, monitor_dir, i, seed) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, training=training, norm_obs=True,
                        norm_reward=norm_reward and training,
                        clip_obs=10.0, gamma=0.99)


def build_model(algo: str, venv: VecNormalize, args, device: str):
    if algo == "ppo":
        n_steps = 1024
        batch = args.n_envs * n_steps // 4
        return PPO(
            "MlpPolicy", venv,
            learning_rate=get_linear_fn(args.lr, args.lr * 0.1, 1.0),
            n_steps=n_steps,
            batch_size=batch,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                               activation_fn=torch.nn.Tanh,
                               log_std_init=-0.5),
            seed=args.seed,
            device=device,
        )
    return SAC(
        "MlpPolicy", venv,
        learning_rate=args.lr,
        buffer_size=1_000_000,
        batch_size=512,
        learning_starts=10_000,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=args.n_envs,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=args.seed,
        device=device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=list(ALGOS), default="ppo")
    ap.add_argument("--run", default=None,
                    help="run directory (default: runs/<algo>_<timestamp>)")
    ap.add_argument("--timesteps", type=int, default=25_000_000)
    ap.add_argument("--n-envs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="'cuda', 'cpu' or 'auto'")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-speed", type=float, default=0.8)
    ap.add_argument("--eval-freq", type=int, default=250_000,
                    help="evaluate every N global steps")
    ap.add_argument("--ckpt-freq", type=int, default=1_000_000,
                    help="checkpoint every N global steps")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the run's last checkpoint")
    args = ap.parse_args()

    run_dir = Path(args.run
                   or f"runs/{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}")
    for sub in ("checkpoints", "monitor", "eval", "logs", "best"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    cfg = WalkConfig(target_speed=args.target_speed)
    (run_dir / "config.json").write_text(json.dumps(
        {"cfg": cfg.to_dict(), "args": vars(args)}, indent=2))

    norm_reward = args.algo == "ppo"
    venv = build_vec_env(cfg, args.n_envs, args.seed,
                         run_dir / "monitor", True, norm_reward)
    eval_env = build_vec_env(cfg, 1, args.seed + 10_000, None, False,
                             norm_reward)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"[train] algo={args.algo.upper()} device={device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})"
          f"  n_envs={args.n_envs}  run={run_dir}")

    if args.resume:
        ckpts = sorted(
            (run_dir / "checkpoints").glob(f"{args.algo}_*_steps.zip"),
            key=lambda p: int(p.stem.split("_")[1]))
        assert ckpts, f"no checkpoint to resume from in {run_dir}/checkpoints"
        vec_pkl = ckpts[-1].with_name(ckpts[-1].stem.replace(
            f"{args.algo}_", f"{args.algo}_vecnormalize_") + ".pkl")
        venv = VecNormalize.load(str(vec_pkl), venv.venv)
        venv.training = True
        model = ALGOS[args.algo].load(str(ckpts[-1]), env=venv, device=device)
        print(f"[train] resumed from {ckpts[-1].name}")
    else:
        model = build_model(args.algo, venv, args, device)
    model.set_logger(configure(str(run_dir / "logs"),
                               ["stdout", "csv", "tensorboard"]))

    callbacks = [
        CheckpointCallback(save_freq=max(args.ckpt_freq // args.n_envs, 1),
                           save_path=str(run_dir / "checkpoints"),
                           name_prefix=args.algo, save_vecnormalize=True),
        EvalCallback(eval_env,
                     best_model_save_path=str(run_dir / "best"),
                     log_path=str(run_dir / "eval"),
                     eval_freq=max(args.eval_freq // args.n_envs, 1),
                     n_eval_episodes=5, deterministic=True),
    ]

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    progress_bar=False, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print("[train] interrupted — saving final state anyway")
    finally:
        model.save(run_dir / "final_model")
        venv.save(str(run_dir / "vecnormalize.pkl"))
        dt = time.time() - t0
        print(f"[train] saved {run_dir}/final_model.zip after "
              f"{model.num_timesteps:,} steps "
              f"({model.num_timesteps / max(dt, 1):,.0f} steps/s, "
              f"{dt / 3600:.2f} h)")
        venv.close()
        eval_env.close()


if __name__ == "__main__":
    main()
