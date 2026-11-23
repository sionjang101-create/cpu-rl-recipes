"""Shared evaluation: 100 episodes, deterministic actions, fixed eval seeds, raw environment reward."""
import gymnasium as gym
import numpy as np

EVAL_EPISODES = 100
EVAL_SEED0 = 10_000


def normalize(obs, obs_rms):
    if obs_rms is None:
        return obs
    return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)


def evaluate(model, env_id, obs_rms=None, episodes=EVAL_EPISODES, seed0=EVAL_SEED0):
    env = gym.make(env_id)
    scores, lengths = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        done, total, steps = False, 0.0, 0
        while not done:
            action, _ = model.predict(normalize(obs, obs_rms), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        scores.append(float(total))
        lengths.append(steps)
    env.close()
    return scores, lengths


def summarize(scores, lengths, recipe):
    out = {"mean": round(float(np.mean(scores)), 2), "std": round(float(np.std(scores)), 2),
           "min": round(float(np.min(scores)), 2), "max": round(float(np.max(scores)), 2),
           "mean_length": round(float(np.mean(lengths)), 1)}
    if recipe.get("full_length"):
        out["full_length_ratio"] = round(sum(l >= recipe["full_length"] for l in lengths) / len(lengths), 3)
    if recipe.get("solved_score") is not None:
        out["solved_ratio"] = round(sum(s >= recipe["solved_score"] for s in scores) / len(scores), 3)
    return out
