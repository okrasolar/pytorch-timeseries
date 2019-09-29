from pathlib import Path

import sys
sys.path.append('..')

from inception import InceptionModel, Trainer


def train_ecg():

    data_folder = Path('../data')

    model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
                           bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
                           num_pred_classes=1)

    trainer = Trainer(model=model, experiment='ECG200', data_folder=data_folder)
    trainer.fit()


if __name__ == '__main__':
    train_ecg()
