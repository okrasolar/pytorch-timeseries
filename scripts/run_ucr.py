"""
Example scripts demonstrating how the UCRTrainer, which extends the BaseTrainer,
can be used to train an Inception Model on UCR Archive data.

The ECG200 dataset has 1 output class, while the Synthetic Control dataset has
6 - in the case of ECG, a sigmoid function is used as the final activation function.
For the Synthetic Control dataset, softmax is used instead.
"""
from pathlib import Path
import sys
sys.path.append('..')

from src import UCRTrainer, load_ucr_trainer
from src.models import InceptionModel, LinearBaseline


def train_inception_ecg():

    data_folder = Path('../data')

    model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
                           bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=1)

    trainer = UCRTrainer(model=model, experiment='ECG200', data_folder=data_folder)
    trainer.fit()

    savepath = trainer.save_model()
    new_trainer = load_ucr_trainer(savepath)
    new_trainer.evaluate()


def train_linear_ecg():

    data_folder = Path('../data')

    model = LinearBaseline(num_inputs=96, num_pred_classes=1)

    trainer = UCRTrainer(model=model, experiment='ECG200', data_folder=data_folder)
    trainer.fit()

    savepath = trainer.save_model()
    new_trainer = load_ucr_trainer(savepath)
    new_trainer.evaluate()


def train_inception_sc():

    data_folder = Path('../data')

    model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
                           bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=6)

    trainer = UCRTrainer(model=model, experiment='synthetic_control', data_folder=data_folder)
    trainer.fit()

    savepath = trainer.save_model()
    new_trainer = load_ucr_trainer(savepath)
    new_trainer.evaluate()


if __name__ == '__main__':
    train_linear_ecg()
