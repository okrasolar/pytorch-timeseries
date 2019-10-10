import torch
from torch import nn


class LinearBaseline(nn.Module):
    """A PyTorch implementation of the Linear Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, num_inputs: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_inputs': num_inputs,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(
            nn.Dropout(0.1),
            LinearBlock(num_inputs, 500, 0.2),
            LinearBlock(500, 500, 0.2),
            LinearBlock(500, num_pred_classes, 0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layers(x.view(x.shape[0], -1))


class LinearBlock(nn.Module):

    def __init__(self, input_size: int, output_size: int,
                 dropout: float) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)
