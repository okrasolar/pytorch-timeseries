from pathlib import Path
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import List, Tuple

from .data import InputData, load_ucr_data, UCR_DATASETS


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

    def _load_experiment(self) -> Tuple[InputData, InputData]:
        assert self.experiment in UCR_DATASETS, \
            f'{self.experiment} must be one of the UCR datasets: ' \
            f'https://www.cs.ucr.edu/~eamonn/time_series_data/'
        experiment_datapath = self.data_folder / 'UCR_TS_Archive_2015' / self.experiment
        return load_ucr_data(experiment_datapath)

    def fit(self, batch_size: int = 64, num_epochs: int = 100,
            val_size: float = 0.2, learning_rate: float = 0.01,
            patience: int = 10) -> None:
        train_data, test_data = self._load_experiment()

        train_data, val_data = train_data.split(val_size)

        train_loader = DataLoader(
            TensorDataset(train_data.x, train_data.y),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_data.x, val_data.y),
            batch_size=batch_size,
            shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_loss = np.inf
        patience_counter = 0

        self.model.train()
        for epoch in range(num_epochs):
            epoch_train_loss = []
            for x_t, y_t in train_loader:
                optimizer.zero_grad()
                output = self.model(x_t)
                if len(y_t.shape) == 1:
                    loss = F.binary_cross_entropy_with_logits(
                        output, y_t.unsqueeze(-1).float(), reduction='mean'
                    )
                else:
                    loss = F.cross_entropy(output, y_t.long(), reduction='mean')

                epoch_train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(epoch_train_loss))

            epoch_val_loss = []
            self.model.eval()
            for x_v, y_v in val_loader:
                with torch.no_grad():
                    output = self.model(x_v)
                    if len(y_v.shape) == 1:
                        loss = F.binary_cross_entropy_with_logits(
                            output, y_v.unsqueeze(-1).float(), reduction='mean'
                        ).item()
                    else:
                        loss = F.cross_entropy(output, y_v.long(), reduction='mean').item()
                    epoch_val_loss.append(loss)
            self.val_loss.append(np.mean(epoch_val_loss))

            print(f'Epoch: {epoch + 1}, '
                  f'Train loss: {round(self.train_loss[-1], 3)}, '
                  f'Val loss: {round(self.val_loss[-1], 3)}')

            if self.val_loss[-1] < best_val_loss:
                best_val_loss = self.val_loss[-1]
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == patience:
                    print('Early stopping!')
                    return None
