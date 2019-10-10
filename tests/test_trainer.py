from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import pytest

from src.models import InceptionModel
from src.trainer import BaseTrainer

from typing import Optional, Tuple


class TrainerForTests(BaseTrainer):
    """A complete trainer class, to test the base trainer's
    functions
    """
    def __init__(self, model: nn.Module, data_folder: Path, in_channels: int,
                 num_preds: int, num_instances: int = 500):
        self.model = model
        self.data_folder = data_folder
        self.model_dir = data_folder / 'model'
        self.model_dir.mkdir()

        self.in_channels = in_channels
        self.num_preds = num_preds
        self.num_instances = num_instances

    def get_loaders(self, batch_size: int, mode: str,
                    val_size: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:

        x = torch.ones(self.num_instances, self.in_channels, 70)
        if self.num_preds > 1:
            y = F.one_hot(torch.randint(size=(self.num_instances, ),
                                        low=0, high=self.num_preds))
        else:
            y = torch.randint(size=(self.num_instances, ), low=0, high=self.num_preds + 1)

        if mode == 'train':
            return (DataLoader(TensorDataset(x, y), batch_size=batch_size),
                    DataLoader(TensorDataset(x, y), batch_size=batch_size))
        else:
            return DataLoader(TensorDataset(x, y), batch_size=batch_size), None


class TestBaseTrainer:

    @pytest.mark.parametrize('num_pred_classes', list(range(1, 5)))
    def test_fit_works_for_single_and_multiclass(self, tmp_path, num_pred_classes):
        in_channels = 30

        model = InceptionModel(2, in_channels, out_channels=30,
                               bottleneck_channels=12, kernel_sizes=15,
                               use_residuals=True, num_pred_classes=num_pred_classes)

        trainer = TrainerForTests(model, tmp_path, in_channels, num_preds=num_pred_classes)
        # this just ensures everything runs
        trainer.fit(batch_size=50, num_epochs=1)
