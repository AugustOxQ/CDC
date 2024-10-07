from email.mime import image
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm_features(image_features: Tensor, text_features: Tensor) -> Tuple[Tensor, Tensor]:
    """Normalize image and text features to unit length.

    Args:
        image_features: Tensor of shape (batch_size, image_feature_dim)
        text_features: Tensor of shape (batch_size, text_feature_dim)

    Returns:
        image_features: Normalized image features. Same shape as input.
        text_features: Normalized text features. Same shape as input.
    """
    norm = torch.norm(image_features, dim=-1, p=2, keepdim=True)
    image_features = torch.div(image_features, norm)
    norm = torch.norm(text_features, dim=-1, p=2, keepdim=True)
    text_features = torch.div(text_features, norm)

    return image_features, text_features


def cross_entropy(preds: Tensor, targets: Tensor, reduction: str = "none") -> torch.Tensor:
    """Computes the cross entropy loss between the input predictions and targets.

    Args:
        preds: The input predictions. A tensor of shape (batch_size, num_classes).
        targets: The target labels. A tensor of shape (batch_size, num_classes).
        reduction: The reduction to apply to the loss. One of "none" or "mean". Defaults to "none".

    Returns:
        The computed loss. A tensor of shape (batch_size,) if reduction is "none", otherwise a scalar.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "mean":
        return loss.mean()
    else:
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        """Initialize ContrastiveLoss module.

        Args:
            margin: Margin value for contrastive loss. Defaults to 1.0.
        """
        super().__init__()
        print("Using ContrastiveLoss")
        self.margin = margin

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        device: torch.device = device,
    ) -> Tensor:
        labels_matrix = torch.eye(image_features.size(0)).to(device)
        label = 1 - labels_matrix

        distances = F.pairwise_distance(image_features, text_features)
        loss = 0.5 * (
            (1 - label) * distances.pow(2) + label * F.relu(self.margin - distances).pow(2)
        )
        return loss.mean()


class CosineLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        """Initialize CosineLoss module.

        Args:
            margin: Margin value for cosine embedding loss. Defaults to 0.1.
        """
        super().__init__()
        print("Using CosineLoss")
        self.margin = margin
        self.loss = nn.CosineEmbeddingLoss(margin=margin)

    def forward(
        self,
        image_features: Tensor,
        combined_features: Tensor,
        device: torch.device = device,
    ) -> Tensor:
        target = torch.ones(image_features.size(0)).to(device)
        output = self.loss(image_features, combined_features, target)

        return output


class LabelContrastiveLoss(nn.Module):
    def __init__(
        self, margin: float = 0.1, reg_weight: float = 1, return_dict: bool = False
    ) -> None:
        """Initialize Combined Cosine and Contrastive Loss module. Cosine Loss will be used to
        contrast combined features and image features. Contrastive Loss will be used to contrast
        positive label and negative label.

        Args:
            margin: Margin value for cosine embedding loss. Defaults to 0.1.
            reg_weight: Weight for regularizer loss. Defaults to 1.
            return_dict: Return loss dictionary or not. Defaults to False.
        """
        super().__init__()
        print("Using Combined Cosine and Contrastive Loss")
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)
        self.reg_weight = reg_weight
        self.return_dict = return_dict
        # TODO Add diversity loss to encourage more diversity in the embeddings

    def forward(
        self,
        image_features: Tensor,
        combined_features: Tensor,
        combined_features_neg: Optional[Tensor] = None,
        device: torch.device = device,
    ):
        # Batch size
        batch_size = image_features.size(0)

        # 1. Cosine loss between image_features and combined_features (positive pair)
        target_positive = torch.ones(batch_size).to(device)
        cosine_loss = self.cosine_loss(image_features, combined_features, target_positive)

        # 2. Contrastive loss with positive and negative contrast if combined_features_neg is provided
        contrastive_loss = 0
        if combined_features_neg is not None:
            # Positive contrast (diagonal should be similar)
            positive_similarity = self.cosine_similarity(image_features, combined_features)
            positive_loss = self.cosine_loss(image_features, combined_features, target_positive)

            # Negative contrast (diagonal should be dissimilar)
            target_negative = -torch.ones(batch_size).to(device)
            negative_similarity = self.cosine_similarity(image_features, combined_features_neg)
            negative_loss = self.cosine_loss(
                image_features, combined_features_neg, target_negative
            )

            # Combine both positive and negative loss for contrastive learning
            contrastive_loss = positive_loss + negative_loss

        # Total loss = cosine loss + contrastive loss (if applicable)
        total_loss = cosine_loss + self.reg_weight * contrastive_loss
        loss_dict = {
            "cosine_loss": cosine_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": total_loss,
        }
        if self.return_dict:
            return loss_dict
        else:
            return total_loss


# TODO: Implement a classifier that choose labels during training, for example, to do concatenation of image, text and label embeddings.


def main():
    ...


if __name__ == "__main__":
    main()
