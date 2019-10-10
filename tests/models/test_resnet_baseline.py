import torch

from src.models import ResNetBaseline


class TestResNetModel:

    def test_model_works_works(self):
        batch_size, sequence_length, in_channels, pred_classes = 100, 50, 32, 12
        input_tensor = torch.ones(batch_size, in_channels, sequence_length)

        model = ResNetBaseline(in_channels=in_channels,
                               num_pred_classes=pred_classes)

        preds = model(input_tensor)
        assert preds.shape == (batch_size, pred_classes)
