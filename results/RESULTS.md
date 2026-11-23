# Results

Protocol for every number below:
- 100 evaluation episodes per seed
- Deterministic actions
- Eval seeds 10000–10099
- Raw environment reward

All runs were trained on one consumer PC, CPU only.

## LunarLander-v3 (PPO, 2M steps, 16 envs)

| Seed | Mean | Std | Min | Landed (≥200) | Train time |
| --- | --- | --- | --- | --- | --- |
| 0 | 267.51 | 20.53 | 227.72 | 100% | 24.3 min |
| 1 | 274.83 | 19.90 | 235.64 | 100% | 26.4 min |
| 2 | 272.70 | 22.30 | 217.98 | 100% | 33.1 min |
| **Mean of 3** | **271.68** | | | | |

- Model: [SionJang/ppo-LunarLander-v3](https://huggingface.co/SionJang/ppo-LunarLander-v3) (seed 1).
- These runs used stable-baselines3 2.9.2.
- With 1M steps and a constant learning rate, the same settings averaged below 260. Doubling the steps and decaying the learning rate linearly to 0 got all three seeds above 260.

## Hopper-v5 (PPO, 1M steps, 1 env, normalized)

| Seed | Mean | Std | Min | Full 1000 steps | Train time |
| --- | --- | --- | --- | --- | --- |
| 0 | 3582.60 | 2.26 | 3577.89 | 100% | 26.0 min |
| 1 | 3502.37 | 153.67 | 1973.94 | 99% | 26.0 min |
| 2 | 3541.83 | 3.16 | 3531.54 | 100% | 26.0 min |
| **Mean of 3** | **3542.27** | | | | |

- Model: [SionJang/ppo-Hopper-v5](https://huggingface.co/SionJang/ppo-Hopper-v5) (seed 0).
- Three seeds ran in parallel, two torch threads each.
- **Reproduction check:** seed 0 was retrained from scratch with this repository's `train.py` (`--seed 0 --threads 2`). All 100 evaluation scores came out identical to the published run: mean 3582.60, std 2.26. Only the wall-clock time differed (59 min, because the PC was busy with 10 other runs).

## Walker2d-v5: Hopper's recipe does not transfer as-is; a wider network helped

First, Hopper's recipe was used unchanged (64x64 network). Then it was compared with a 256x256 network. Nothing else changed.

| Network | Seeds | Failed seeds (mean < 1500) | Mean of all seeds | Median | Worst seed |
| --- | --- | --- | --- | --- | --- |
| 64x64 (Hopper unchanged) | 8 | **1** | 3840.79 | 4204.84 | 1072.68 |
| **256x256 (recipe default)** | 5 | **0** | **4918.22** | 4844.09 | 4380.69 |

Per-seed means:
- 64x64: 1072.68, 4510.33, 3946.78, 5081.36, 4124.83, 4284.85, 3414.76, 4290.72
- 256x256: 4905.96, 4844.09, 5744.13, 4716.22, 4380.69

- Model: [SionJang/ppo-Walker2d-v5](https://huggingface.co/SionJang/ppo-Walker2d-v5) (256x256, seed 0: mean 4905.96, 100/100 episodes reach 1000 steps).

What the failed seed did:
- Seed 0 with 64x64 lunges forward instead of walking and falls at around step 330. It does this in all 100 episodes.
- The same seed with 256x256 averages 4905.96 and finishes all 100 episodes.

What we can and cannot say:
- **Scores:** the 256x256 improvement is consistent. Its worst seed is above the 64x64 median.
- **Failure rate:** 0/5 vs 1/8 is not statistically distinguishable. Failures are rare, so comparing failure rates properly needs 20+ seeds per setting.
- **Run conditions:**
  - 64x64 seeds 0–2 ran three at a time with 2 torch threads each.
  - The other 10 runs ran together with 1 thread each, 60–69 min per seed.
  - Alone, a seed takes about 30 min.

## What we do not claim

- We have not compared these numbers with other public models under the same protocol. Scores from different evaluation setups are not comparable.
- Some results here come from three to five seeds only.
