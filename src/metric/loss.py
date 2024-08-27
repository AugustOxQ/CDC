from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm_features(image_features, text_features):
    norm = torch.norm(image_features, dim=-1, p=2, keepdim=True)
    image_features = torch.div(image_features, norm)
    norm = torch.norm(text_features, dim=-1, p=2, keepdim=True)
    text_features = torch.div(text_features, norm)

    return image_features, text_features


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        print("Using ContrastiveLoss")
        self.margin = margin

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        device: torch.device = device,
    ):
        labels_matrix = torch.eye(image_features.size(0)).to(device)
        label = 1 - labels_matrix
        
        distances = F.pairwise_distance(image_features, text_features)
        loss = 0.5 * ((1 - label) * distances.pow(2) + 
                      label * F.relu(self.margin - distances).pow(2))
        return loss.mean()


class CosineLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        print("Using CosineLoss")
        self.margin = margin
        self.loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        device: torch.device = device,
    ):
        # image_features, text_features = norm_features(image_features, text_features)
        target = torch.ones(image_features.size(0)).to(device)
        output = self.loss(image_features, text_features, target)

        return output

class MeanSquareLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        print("Using MeanSquareLoss")
        self.loss = nn.MSELoss()

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        device: torch.device = device,
    ):
        # image_features, text_features = norm_features(image_features, text_features)
        output = self.loss(image_features, text_features)
        return output


# TODO: Implement a classifier that choose labels during training, for example, to do concatenation of image, text and label embeddings.

def main():
    cosineloss = CosineLoss()

    image_features = torch.randn(128, 512)
    text_features = torch.randn(128, 512)

    loss = cosineloss(image_features, text_features, device="cpu")

    print("Contrastive Loss: ")
    print(loss)

    mse_loss = MeanSquareLoss()
    loss = mse_loss(image_features, text_features, device="cpu")

    print("Mean Square Loss: ")
    print(loss)


if __name__ == "__main__":
    main()
