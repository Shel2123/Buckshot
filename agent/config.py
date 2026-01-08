import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    total_timesteps_per_generation: int = 100_000
    learning_rate: float = 1e-4
    n_steps: int = 4096
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    n_envs: int = 16

    # Evaluation Settings
    eval_random_episodes: int = 2500
    eval_champion_episodes: int = 25000
    win_threshold: float = 0.5025  # 50.25% required to become the new king.
    use_paired_evaluation: bool = True  # Use Common Random Numbers (CRN)

    # Opponent Pool Settings
    pool_size: int = 10

    # Paths
    models_dir: str = "agent/models"
    champions_dir: str = "agent/models/champions"

    # Global Seed
    seed: int = 42


def set_global_seed(seed: int, configure_cudnn: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if configure_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
