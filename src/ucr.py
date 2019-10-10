from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import models
from .trainer import BaseTrainer

from typing import Dict, List, Tuple, Optional


@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float) -> Tuple[InputData, InputData]:
        train_x, val_x, train_y, val_y = train_test_split(
            self.x.numpy(), self.y.numpy(), test_size=split_size, stratify=self.y
        )
        return (InputData(x=torch.from_numpy(train_x), y=torch.from_numpy(train_y)),
                InputData(x=torch.from_numpy(val_x), y=torch.from_numpy(val_y)))


UCR_DATASETS = ['Haptics', 'Worms', 'Computers', 'UWaveGestureLibraryAll',
                'Strawberry', 'Car', 'BeetleFly', 'wafer', 'CBF', 'Adiac',
                'Lighting2', 'ItalyPowerDemand', 'yoga', 'Trace', 'ShapesAll',
                'Beef', 'MALLAT', 'MiddlePhalanxTW', 'Meat', 'Herring',
                'MiddlePhalanxOutlineCorrect', 'FordA', 'SwedishLeaf',
                'SonyAIBORobotSurface', 'InlineSkate', 'WormsTwoClass', 'OSULeaf',
                'Ham', 'uWaveGestureLibrary_Z', 'NonInvasiveFatalECG_Thorax1',
                'ToeSegmentation1', 'ScreenType', 'SmallKitchenAppliances',
                'WordsSynonyms', 'MoteStrain', 'synthetic_control', 'Cricket_X',
                'ECGFiveDays', 'Wine', 'Cricket_Y', 'TwoLeadECG', 'Two_Patterns',
                'Phoneme', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
                'DistalPhalanxTW', 'FacesUCR', 'ECG5000', '50words', 'HandOutlines',
                'Coffee', 'Gun_Point', 'FordB', 'InsectWingbeatSound', 'MedicalImages',
                'Symbols', 'ArrowHead', 'ProximalPhalanxOutlineAgeGroup',
                'SonyAIBORobotSurfaceII', 'ChlorineConcentration', 'Plane', 'Lighting7',
                'PhalangesOutlinesCorrect', 'ShapeletSim', 'DistalPhalanxOutlineAgeGroup',
                'uWaveGestureLibrary_X', 'FaceFour', 'RefrigerationDevices', 'ECG200',
                'ToeSegmentation2', 'CinC_ECG_torso', 'BirdChicken', 'OliveOil',
                'LargeKitchenAppliances', 'uWaveGestureLibrary_Y',
                'NonInvasiveFatalECG_Thorax2', 'FISH', 'ProximalPhalanxOutlineCorrect',
                'Cricket_Z', 'FaceAll', 'StarLightCurves', 'ElectricDevices', 'Earthquakes',
                'DiatomSizeReduction', 'ProximalPhalanxTW']


def load_ucr_data(data_path: Path,
                  encoder: Optional[OneHotEncoder] = None
                  ) -> Tuple[InputData, InputData, OneHotEncoder]:

    experiment = data_path.parts[-1]

    train = np.loadtxt(data_path / f'{experiment}_TRAIN', delimiter=',')
    test = np.loadtxt(data_path / f'{experiment}_TEST', delimiter=',')

    if encoder is None:
        encoder = OneHotEncoder(categories='auto', sparse=False)
        y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
    else:
        y_train = encoder.transform(np.expand_dims(train[:, 0], axis=-1))
    y_test = encoder.transform(np.expand_dims(test[:, 0], axis=-1))

    if y_train.shape[1] == 2:
        # there are only 2 classes, so there only needs to be one
        # output
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

    # UCR data is univariate, so an additional dimension is added
    # at index 1 to make it of shape (N, Channels, Length)
    # as the model expects
    train_input = InputData(x=torch.from_numpy(train[:, 1:]).unsqueeze(1).float(),
                            y=torch.from_numpy(y_train))
    test_input = InputData(x=torch.from_numpy(test[:, 1:]).unsqueeze(1).float(),
                           y=torch.from_numpy(y_test))
    return train_input, test_input, encoder


class UCRTrainer(BaseTrainer):
    """Train the model on UCR datasets

    Attributes
    ----------
    model:
        The initialized inception model
    experiment:
        The UCR/UEA dataset to train the model on
    data_folder:
        The location of the data_folder
    """

    def __init__(self, model: nn.Module, experiment: str,
                 data_folder: Path = Path('data')) -> None:
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
        train_data, test_data = self._load_data()

        if mode == 'train':
            assert val_size is not None, 'Val size must be defined when loading training data'
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

            return train_loader, val_loader
        else:
            test_loader = DataLoader(
                TensorDataset(test_data.x, test_data.y),
                batch_size=batch_size,
                shuffle=False,
            )
            return test_loader, None

    def save_model(self, savepath: Optional[Path] = None) -> Path:
        save_dict = {
            'model': {
                'model_class': self.model.__class__.__name__,
                'state_dict': self.model.state_dict(),
                'input_args': self.model.input_args,
            },
            'encoder': self.encoder
        }
        if savepath is None:
            model_name = f'{self.model.__class__.__name__}_{self.experiment}_model.pkl'
            savepath = self.model_dir / model_name
        torch.save(save_dict, savepath)

        return savepath


def load_ucr_trainer(model_path: Path) -> UCRTrainer:

    experiment = model_path.resolve().parts[-2]
    data_folder = model_path.resolve().parents[3]

    model_dict = torch.load(model_path)

    model_class = getattr(models, model_dict['model']['model_class'])
    model = model_class(**model_dict['model']['input_args'])
    model.load_state_dict(model_dict['model']['state_dict'])

    loaded_trainer = UCRTrainer(model, experiment=experiment,
                                data_folder=data_folder)
    loaded_trainer.encoder = model_dict['encoder']

    return loaded_trainer
