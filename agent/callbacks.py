from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class GenerationCallback(BaseCallback):
    """Callback to log generation-specific metrics."""

    def __init__(self, generation: int, verbose=0):
        super().__init__(verbose)
        self.generation = generation

    def _on_step(self) -> bool:
        return True


class ProgressCallback(BaseCallback):
    """
    Callback to update a TQDM progress bar during training.
    """

    def __init__(self, pbar: tqdm, initial_timesteps: int):
        super().__init__()
        self.pbar = pbar
        self.initial_timesteps = initial_timesteps
        self.last_reported = 0

    def _on_step(self) -> bool:
        # Calculate progress within this specific generation
        current_progress = self.model.num_timesteps - self.initial_timesteps
        delta = current_progress - self.last_reported

        if delta > 0:
            self.pbar.update(delta)
            self.last_reported = current_progress

        return True
