"""Train one seed of a recipe and evaluate it.

    python train.py --env Hopper-v5 --seed 0
    python train.py --env Walker2d-v5 --seed 3 --net 256      # override the network width
    python train.py --env Hopper-v5 --seed 0 --steps 20000    # quick smoke test

Runs on CPU. To run several seeds at once, start one process per seed.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from common import evaluate, summarize
from recipes import RECIPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(RECIPES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--net", type=int, help="hidden layer width for policy and value networks")
    ap.add_argument("--steps", type=int, help="override total training steps")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per process")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    recipe = dict(RECIPES[a.env])
    width = a.net or recipe["net"]
    steps = a.steps or recipe["steps"]
    torch.set_num_threads(a.threads)
    out = Path(a.out) / f"{a.env}_net{width}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = make_vec_env(a.env, n_envs=recipe["n_envs"], seed=a.seed)
    if recipe["normalize"]:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    lr0 = recipe["lr"]
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=0, device="cpu", learning_rate=lambda p: lr0 * p,
                policy_kwargs=dict(net_arch=dict(pi=[width, width], vf=[width, width])), **recipe["ppo"])
    model.learn(total_timesteps=steps)
    train_min = (time.time() - t0) / 60

    name = f"ppo_seed{a.seed}"
    model.save(out / name)
    obs_rms = None
    if recipe["normalize"]:
        env.save(str(out / f"vecnormalize_seed{a.seed}.pkl"))
        obs_rms = env.obs_rms
    scores, lengths = evaluate(model, a.env, obs_rms, episodes=a.eval_episodes)
    result = {"env": a.env, "seed": a.seed, "net": width, "steps": int(model.num_timesteps),
              "train_minutes": round(train_min, 1), **summarize(scores, lengths, recipe),
              "scores": [round(s, 2) for s in scores]}
    (out / f"result_seed{a.seed}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}), flush=True)


if __name__ == "__main__":
    main()
