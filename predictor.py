import os

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as T
from .helpers import get_data_location


class Predictor(nn.Module):
    def __init__(self, model, class_names, mean, std):
        super().__init__()
        self.model = model.eval()
        self.class_names = class_names

        # Save mean/std as parameters for scripting compatibility
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to (256, 256) using interpolate
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        # Center crop to (224, 224)
        crop_size = 224
        _, _, h, w = x.shape
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        x = x[:, :, top:top+crop_size, left:left+crop_size]

        # Convert to float if not already (optional)
        if not torch.is_floating_point(x):
            x = x.float()

        # Normalize
        x = (x - self.mean) / self.std

        # Forward through model
        x = self.model(x)

        # Softmax
        x = F.softmax(x, dim=1)
        return x


def predictor_test(test_dataloader, model_reloaded):
    """
    Test the predictor. Since the predictor does not operate on the same tensors
    as the non-wrapped model, we need a specific test function (can't use one_epoch_test)
    """

    folder = get_data_location()
    test_data = datasets.ImageFolder(os.path.join(folder, "test"), transform=T.ToTensor())

    pred = []
    truth = []
    for x in tqdm(test_data, total=len(test_dataloader.dataset), leave=True, ncols=80):
        softmax = model_reloaded(x[0].unsqueeze(dim=0))

        idx = softmax.squeeze().argmax()

        pred.append(int(x[1]))
        truth.append(int(idx))

    pred = np.array(pred)
    truth = np.array(truth)

    print(f"Accuracy: {(pred==truth).sum() / pred.shape[0]}")

    return truth, pred


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    from .model import MyModel
    from .helpers import compute_mean_and_std

    mean, std = compute_mean_and_std()

    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    predictor = Predictor(model, class_names=['a', 'b', 'c'], mean=mean, std=std)

    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(),
        torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
