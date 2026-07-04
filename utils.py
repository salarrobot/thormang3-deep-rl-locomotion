from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import ThormangWalkEnv, WalkConfig

ALGOS = {"ppo": PPO, "sac": SAC}


def resolve_model(run_dir: Path) -> tuple[Path, Path]:
    best = run_dir / "best/best_model.zip"
    final = run_dir / "final_model.zip"
    vec = run_dir / "vecnormalize.pkl"
    if best.exists() and vec.exists():
        return best, vec
    if final.exists() and vec.exists():
        return final, vec
    ckpts = sorted((run_dir / "checkpoints").glob("*_steps.zip"),
                   key=lambda p: int(p.stem.split("_")[-2]))
    assert ckpts, f"no model found in {run_dir}"
    ck = ckpts[-1]
    algo = ck.stem.split("_")[0]
    vec = ck.with_name(
        ck.stem.replace(f"{algo}_", f"{algo}_vecnormalize_") + ".pkl")
    return ck, vec


def run_algo(run_dir: Path) -> str:
    cfg = run_dir / "config.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["args"].get("algo", "ppo")
    return "ppo"


def run_config(run_dir: Path | None) -> WalkConfig:
    if run_dir and (run_dir / "config.json").exists():
        d = json.loads((run_dir / "config.json").read_text())["cfg"]
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return WalkConfig(**d)
    return WalkConfig()


def load_model(path: Path, algo: str | None = None,
               device: str = "cpu") -> BaseAlgorithm:
    if algo in ALGOS:
        return ALGOS[algo].load(str(path), device=device)
    last_err = None
    for cls in ALGOS.values():
        try:
            return cls.load(str(path), device=device)
        except Exception as e:
            last_err = e
    raise last_err


def load_runner(model_path: Path, vec_path: Path, cfg: WalkConfig,
                algo: str | None = None, seed: int = 7,
                **env_kwargs) -> tuple[BaseAlgorithm, VecNormalize,
                                       ThormangWalkEnv]:
    venv = VecNormalize.load(
        str(vec_path),
        DummyVecEnv([lambda: ThormangWalkEnv(cfg=cfg, **env_kwargs)]))
    venv.training = False
    venv.norm_reward = False
    venv.seed(seed)
    model = load_model(model_path, algo)
    raw: ThormangWalkEnv = venv.venv.envs[0].unwrapped
    return model, venv, raw
