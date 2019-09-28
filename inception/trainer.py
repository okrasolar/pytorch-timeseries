from dataclasses import dataclass

import torch
from torch import nn
from pathlib import Path

from typing import List, Tuple


@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor


class Trainer:

    def __init__(self, model: nn.Module, experiment: str, data_folder: Path = Path('data')) -> None:
        self.model = model
        self.experiment = experiment
        self.data_folder = data_folder

        self.model_dir = data_folder / 'models' / self.model.__class__.__name__ / experiment
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # to be filled by the fit function
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []

    def _load_experiment(self) -> Tuple[InputData, InputData, InputData]:

    def fit(self, batch_size: int = 64, num_epochs: int = 100,
            early_stopping: int = 10) -> None:
        raise NotImplementedError
