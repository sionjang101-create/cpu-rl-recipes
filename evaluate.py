"""Evaluate a saved model with the same protocol as the results table.

    python evaluate.py --env Hopper-v5 --model ppo_seed0.zip --vecnorm vecnormalize_seed0.pkl
"""
import argparse
import json
import pickle

from stable_baselines3 import PPO

from common import evaluate, summarize
from recipes import RECIPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(RECIPES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", help="VecNormalize statistics saved during training (MuJoCo recipes)")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    model = PPO.load(a.model, device="cpu")
    obs_rms = None
    if a.vecnorm:
        with open(a.vecnorm, "rb") as f:  # only load statistics files you trust: pickle can run code
            obs_rms = pickle.load(f).obs_rms
    scores, lengths = evaluate(model, a.env, obs_rms, episodes=a.episodes)
    print(json.dumps(summarize(scores, lengths, RECIPES[a.env])))


if __name__ == "__main__":
    main()
