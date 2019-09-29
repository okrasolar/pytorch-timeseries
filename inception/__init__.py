from pathlib import Path
import torch

from .inception import InceptionModel
from .trainer import Trainer


__all__ = ['InceptionModel', 'Trainer']


def load_trainer(model_path: Path) -> Trainer:

    experiment = model_path.resolve().parts[-2]
    data_folder = model_path.resolve().parents[3]

    model_dict = torch.load(model_path)

    model = InceptionModel(**model_dict['model']['input_args'])
    model.load_state_dict(model_dict['model']['state_dict'])

    loaded_trainer = Trainer(model, experiment=experiment,
                             data_folder=data_folder)
    loaded_trainer.encoder = model_dict['encoder']

    return loaded_trainer
