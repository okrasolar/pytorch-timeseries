from torch import nn
from pathlib import Path

from typing import List, Tuple

from .data import InputData, load_ucr_data, UCR_DATASETS


class Trainer:

    def __init__(self, model: nn.Module, experiment: str, data_folder: Path = Path('data')) -> None:
        self.model = model

        self.experiment = experiment
        assert self.experiment in UCR_DATASETS, \
            f'{experiment} must be one of the UCR datasets: ' \
            f'https://www.cs.ucr.edu/~eamonn/time_series_data/'
        self.data_folder = data_folder

        self.model_dir = data_folder / 'models' / self.model.__class__.__name__ / experiment
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # to be filled by the fit function
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []

    def _load_experiment(self) -> Tuple[InputData, InputData]:
        experiment_datapath = self.data_folder / 'UCR_TS_Archive_2015' / self.experiment
        return load_ucr_data(experiment_datapath)

    def fit(self, batch_size: int = 64, num_epochs: int = 100,
            val_size: float = 0.2, early_stopping: int = 10) -> None:
        train_data, test_data = self._load_experiment()

        print(train_data.y)
