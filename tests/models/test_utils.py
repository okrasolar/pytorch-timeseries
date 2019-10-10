import torch
import pytest

from src.models.utils import Conv1dSamePadding


class TestConv1dSamePadding:

    @pytest.mark.parametrize('kernel,dilation,stride', [(1, 1, 2), (2, 3, 4)])
    def test_same_padding_conserves_length(self, kernel, dilation, stride):

        batch, in_channels, out_channels, length = 100, 32, 38, 50
        input_tensor = torch.ones(batch, in_channels, length)

        model = Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel, dilation=dilation, stride=stride,
                                  bias=False)

        output = model(input_tensor)
        assert output.shape[2] == length
        assert output.shape[1] == out_channels
        assert output.shape[0] == batch
