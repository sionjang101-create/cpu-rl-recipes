# CPU RL Recipes

Reinforcement learning recipes you can reproduce on an ordinary PC, CPU only, in about 30 minutes per run.

Each recipe comes with:
- the exact settings
- a fixed evaluation protocol: 100 episodes, eval seeds 10000–10099
- every seed's result, including the failures

| Environment | Mean of seeds | Worst seed | Status |
| --- | --- | --- | --- |
| LunarLander-v3 | 271.68 | 267.51 | all seeds above 260 |
| Hopper-v5 | 3542.27 | 3502.37 | all seeds finish 99–100% of episodes; seed 0 retrained with this repo gives identical scores |
| Walker2d-v5 | 4918.22 (5 seeds) | 4380.69 | Hopper's recipe unchanged failed 1 seed in 8; a 256x256 network fixed the scores, failure rate not yet proven |

Details and per-seed tables are in [results/RESULTS.md](results/RESULTS.md).

## Why

Most published RL scores show one good run. On a new task, the questions that matter are different:
- How often does a run fail?
- How long does it take on the hardware you actually have?
- Can you get the same number again?

This repository reports all seeds with one evaluation protocol, so you can check our numbers and compare your changes against a fixed baseline.

## Run

```bash
pip install -r requirements.txt
python train.py --env Hopper-v5 --seed 0                  # about 26 min on CPU
python train.py --env Hopper-v5 --seed 0 --steps 20000    # quick smoke test
python evaluate.py --env Hopper-v5 --model runs/Hopper-v5_net64/ppo_seed0.zip --vecnorm runs/Hopper-v5_net64/vecnormalize_seed0.pkl
```

- Run several seeds at once by starting one process per seed.
- MuJoCo recipes save observation-normalization statistics (`vecnormalize_seed*.pkl`). Without them, a trained model will not reach its reported score.
- Load `.pkl` files only from sources you trust. Python pickle can run code.

## Files

| File | Purpose |
| --- | --- |
| `recipes.py` | settings for each environment, exactly as used for the results |
| `train.py` | train one seed, then evaluate it with the fixed protocol |
| `evaluate.py` | evaluate a saved model with the same protocol |
| `common.py` | evaluation and summary code shared by both |

## How this was made

An AI agent, Rich (Claude Code), did the research work:
- designed the experiments
- wrote the code
- ran the training
- evaluated and recorded the results

A human approved each experiment and reviewed the results before release. Failures are published alongside successes.

## License

MIT
