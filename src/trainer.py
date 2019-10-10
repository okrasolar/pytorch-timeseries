from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import cast, Any, Dict, List, Tuple, Optional


class BaseTrainer:
    """Trains an inception model. Dataset-specific trainers should extend this class
    and implement __init__, get_loaders and save functions.
    See UCRTrainer in .ucr.py for an example.

    Attributes
    ----------
    The following need to be added by the initializer:
    model:
        The initialized inception model
    data_folder:
        A path to the data folder - get_loaders should look here for the data
    model_dir:
        A path to where the model and its predictions should be saved

    The following don't:
    train_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    val_loss:
        The fit function fills this list in as the model trains. Useful for plotting
    test_results:
        The evaluate function fills this in, evaluating the model on the test data
    """
    model: nn.Module
    data_folder: Path
    model_dir: Path
    train_loss: List[float] = []
    val_loss: List[float] = []
    test_results: Dict[str, float] = {}
    input_args: Dict[str, Any] = {}

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
        train_loader, val_loader = self.get_loaders(batch_size, mode='train', val_size=val_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_val_loss = np.inf
        patience_counter = 0
        best_state_dict = None

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
            for x_v, y_v in cast(DataLoader, val_loader):
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
                best_state_dict = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter == patience:
                    if best_state_dict is not None:
                        self.model.load_state_dict(cast(Dict[str, torch.Tensor], best_state_dict))
                    print('Early stopping!')
                    return None

    def evaluate(self, batch_size: int = 64) -> None:

        test_loader, _ = self.get_loaders(batch_size, mode='test')

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

    @staticmethod
    def _to_1d_binary(y_true: np.ndarray, y_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(y_true.shape) > 1:
            return np.argmax(y_true, axis=-1), np.argmax(y_preds, axis=-1)

        else:
            return y_true, (y_preds > 0.5).astype(int)

    def get_loaders(self, batch_size: int, mode: str,
                    val_size: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Return dataloaders of the training / test data

        Arguments
        ----------
        batch_size:
            The batch size each iteration of the dataloader should return
        mode: {'train', 'test'}
            If 'train', this function should return (train_loader, val_loader)
            If 'test', it should return (test_loader, None)
        val_size:
            If mode == 'train', the fraction of training data to use for validation
            Ignored if mode == 'test'

        Returns
        ----------
        Tuple of (train_loader, val_loader) if mode == 'train'
        Tuple of (test_loader, None) if mode == 'test'
        """
        raise NotImplementedError

    def save_model(self, savepath: Optional[Path] = None) -> Path:
        raise NotImplementedError
