from pathlib import Path
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Dict, List, Tuple, Optional

from .data import InputData, load_ucr_data, UCR_DATASETS


class Trainer:
    """Trains an inception model

    Attributes
    ----------
    model:
        The initialized inception model
    experiment:
        The UCR/UEA dataset to train the model on
    data_folder:
        The location of the data_folder
    """

    def __init__(self, model: nn.Module, experiment: str, data_folder: Path = Path('data')) -> None:
        self.model = model

        self.experiment = experiment
        self.data_folder = data_folder

        self.model_dir = data_folder / 'models' / self.model.__class__.__name__ / experiment
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # to be filled by the fit function
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.test_results: Dict[str, float] = {}

        self.encoder: Optional[OneHotEncoder] = None

    def _load_data(self) -> Tuple[InputData, InputData]:
        assert self.experiment in UCR_DATASETS, \
            f'{self.experiment} must be one of the UCR datasets: ' \
            f'https://www.cs.ucr.edu/~eamonn/time_series_data/'
        experiment_datapath = self.data_folder / 'UCR_TS_Archive_2015' / self.experiment
        if self.encoder is None:
            train, test, encoder = load_ucr_data(experiment_datapath)
            self.encoder = encoder
        else:
            train, test, _ = load_ucr_data(experiment_datapath, encoder=self.encoder)
        return train, test

    def fit(self, batch_size: int = 64, num_epochs: int = 100,
            val_size: float = 0.2, learning_rate: float = 0.01,
            patience: int = 10) -> None:
        """Trains the inception model

        Arguments
        ----------
        batch_size:
            Batch size to use for training and validation
        num_epochs:
            Maximum number of epochs to train for
        val_size:
            Fraction of training set to use for validation
        learning_rate:
            Learning rate to use with Adam optimizer
        patience:
            Maximum number of epochs to wait without improvement before
            early stopping
        """
        train_data, _ = self._load_data()

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
                    train_loss = F.binary_cross_entropy_with_logits(
                        output, y_t.unsqueeze(-1).float(), reduction='mean'
                    )
                else:
                    train_loss = F.cross_entropy(output, y_t.argmax(dim=-1), reduction='mean')

                epoch_train_loss.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
            self.train_loss.append(np.mean(epoch_train_loss))

            epoch_val_loss = []
            self.model.eval()
            for x_v, y_v in val_loader:
                with torch.no_grad():
                    output = self.model(x_v)
                    if len(y_v.shape) == 1:
                        val_loss = F.binary_cross_entropy_with_logits(
                            output, y_v.unsqueeze(-1).float(), reduction='mean'
                        ).item()
                    else:
                        val_loss = F.cross_entropy(output,
                                                   y_v.argmax(dim=-1), reduction='mean').item()
                    epoch_val_loss.append(val_loss)
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

    def evaluate(self, batch_size: int = 64) -> None:

        _, test = self._load_data()

        test_loader = DataLoader(
            TensorDataset(test.x, test.y),
            batch_size=batch_size,
            shuffle=False,
        )

        self.model.eval()

        true_list, preds_list = [], []
        for x, y in test_loader:
            with torch.no_grad():
                true_list.append(y.detach().numpy())
                preds = self.model(x)
                if len(y.shape) == 1:
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.softmax(preds, dim=-1)
                preds_list.append(preds.detach().numpy())

        true_np, preds_np = np.concatenate(true_list), np.concatenate(preds_list)

        self.test_results['roc_auc_score'] = roc_auc_score(true_np, preds_np)
        print(f'ROC AUC score: {round(self.test_results["roc_auc_score"], 3)}')

        self.test_results['accuracy_score'] = accuracy_score(
            *self._to_1d_binary(true_np, preds_np)
        )
        print(f'Accuracy score: {round(self.test_results["accuracy_score"], 3)}')

    def save_model(self, savepath: Optional[Path] = None) -> Path:
        save_dict = {
            'model': {
                'state_dict': self.model.state_dict(),
                'input_args': self.model.input_args,
            },
            'encoder': self.encoder
        }
        if savepath is None:
            savepath = self.model_dir / 'model.pkl'
        torch.save(save_dict, savepath)

        return savepath

    @staticmethod
    def _to_1d_binary(y_true: np.ndarray, y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(y_true.shape) > 1:
            return np.argmax(y_true, axis=-1), np.argmax(y_preds, axis=-1)

        else:
            return y_true, (y_preds > 0.5).astype(int)
