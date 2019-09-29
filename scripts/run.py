from pathlib import Path

import sys
sys.path.append('..')

from inception import InceptionModel, Trainer, load_trainer


def train_ecg():

    data_folder = Path('../data')

    model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
                           bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=1)

    trainer = Trainer(model=model, experiment='ECG200', data_folder=data_folder)
    trainer.fit()

    savepath = trainer.save_model()
    new_trainer = load_trainer(savepath)
    new_trainer.evaluate()


def train_sc():

    data_folder = Path('../data')

    model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
                           bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=6)

    trainer = Trainer(model=model, experiment='synthetic_control', data_folder=data_folder)
    trainer.fit()

    savepath = trainer.save_model()
    new_trainer = load_trainer(savepath)
    new_trainer.evaluate()


if __name__ == '__main__':
    train_sc()
