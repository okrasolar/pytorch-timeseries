from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from typing import Tuple


@dataclass
class InputData:
    x: torch.Tensor
    y: torch.Tensor

    def split(self, split_size: float) -> Tuple[InputData, InputData]:
        train_x, val_x, train_y, val_y  = train_test_split(
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


def load_ucr_data(data_path: Path) -> Tuple[InputData, InputData]:

    experiment = data_path.parts[-1]

    train = np.loadtxt(data_path / f'{experiment}_TRAIN', delimiter=',')
    test = np.loadtxt(data_path / f'{experiment}_TEST', delimiter=',')

    encoder = OneHotEncoder(categories='auto', sparse=False)
    y_train = encoder.fit_transform(np.expand_dims(train[:, 0], axis=-1))
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
    return train_input, test_input
