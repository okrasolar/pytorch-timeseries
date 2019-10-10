import torch
import pytest

from src.models.inception import InceptionBlock, InceptionModel


class TestInceptionModel:

    def test_model_works_with_scalar_inputs(self):
        batch_size, num_blocks, in_channels, pred_classes = 100, 3, 32, 12

        input_tensor = torch.ones(batch_size, in_channels, 50)

        model = InceptionModel(num_blocks, in_channels, out_channels=30,
                               bottleneck_channels=12, kernel_sizes=15,
                               use_residuals=True, num_pred_classes=pred_classes)

        preds = model(input_tensor)
        assert preds.shape == (batch_size, pred_classes)

    def test_model_works_with_list_inputs(self):
        batch_size, num_blocks, in_channels, pred_classes = 100, 2, 32, 12

        input_tensor = torch.ones(batch_size, in_channels, 50)

        model = InceptionModel(num_blocks, in_channels, out_channels=[30, 25],
                               bottleneck_channels=[12, 0], kernel_sizes=[15, 12],
                               use_residuals=[True, False], num_pred_classes=pred_classes)

        preds = model(input_tensor)
        assert preds.shape == (batch_size, pred_classes)

    def test_model_catches_incorrect_inputs(self):
        num_blocks, in_channels, pred_classes = 2, 32, 12

        with pytest.raises(AssertionError) as e:
            InceptionModel(num_blocks, in_channels, out_channels=[30, 25, 75],
                           bottleneck_channels=[12, 0], kernel_sizes=[15, 12],
                           use_residuals=[True, False], num_pred_classes=pred_classes)

        # make sure we get the right error message
        expected = 'Length of inputs lists must be the same as num blocks, ' \
                   'expected length 2, got 3'
        assert str(e.value) == expected


class TestInceptionBlock:

    @pytest.mark.parametrize('residual,bottleneck,kernel', [(True, 8, 10),
                                                            (False, 0, 24),
                                                            (True, 0, 15),
                                                            (False, 19, 30)])
    def test_inception_block_conserves_length(self, residual, bottleneck, kernel):
        batch, in_channels, out_channels, length = 100, 32, 38, 50
        input_tensor = torch.ones(batch, in_channels, length)

        model = InceptionBlock(in_channels=in_channels, out_channels=out_channels,
                               residual=residual, bottleneck_channels=bottleneck,
                               kernel_size=kernel)
        output = model(input_tensor)
        assert output.shape[2] == length
        assert output.shape[1] == out_channels
        assert output.shape[0] == batch
