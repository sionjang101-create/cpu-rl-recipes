"""Training recipes. Each entry is exactly the setting used for the results in results/RESULTS.md."""

COMMON_PPO = dict(gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5)

RECIPES = {
    # Discrete control. Starting point: RL Baselines3 Zoo LunarLander PPO, plus 2x steps and a linear LR decay.
    "LunarLander-v3": dict(
        steps=2_000_000, n_envs=16, normalize=False, net=64,
        ppo=dict(n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01),
        lr=3e-4,  # decays linearly to 0
        full_length=None, solved_score=200,
    ),
    # Continuous control (MuJoCo). One environment, observation + reward normalization.
    "Hopper-v5": dict(
        steps=1_000_000, n_envs=1, normalize=True, net=64,
        ppo=dict(COMMON_PPO, n_steps=2048, batch_size=64, n_epochs=10),
        lr=3e-4, full_length=1000, solved_score=None,
    ),
    # Hopper's recipe with a wider network. With net=64 (Hopper unchanged), 1 seed in 8 failed; see results.
    "Walker2d-v5": dict(
        steps=1_000_000, n_envs=1, normalize=True, net=256,
        ppo=dict(COMMON_PPO, n_steps=2048, batch_size=64, n_epochs=10),
        lr=3e-4, full_length=1000, solved_score=None,
    ),
}
