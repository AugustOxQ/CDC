from email.mime import image
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.metric.regularizer import (
    angular_consistency_loss,
    boundary_penalty,
    entropy_loss,
    label_change_regularizer,
    pull_away_diversity_loss,
    text_preserve_regularizer,
)

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


def compute_cosine_similarity(features1: Tensor, features2: Tensor) -> Tensor:
    """Compute the pairwise cosine similarity between two sets of feature vectors.

    Args:
        features1: Tensor of shape (batch_size, feature_dim)
        features2: Tensor of shape (batch_size, feature_dim)
    Returns:
        Tensor of shape (batch_size, batch_size) containing pairwise cosine similarities.
    """
    # Normalize the feature vectors
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)

    # Compute the cosine similarity as the dot product of normalized features
    cosine_sim = torch.mm(features1_norm, features2_norm.t())

    return cosine_sim


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
        """
        Compute the contrastive loss between the image and text features.

        Args:
            image_features: The image features. A tensor of shape (batch_size, feature_dim).
            text_features: The text features. A tensor of shape (batch_size, feature_dim).
            device: The device to use for computation. Defaults to the device set by geoopt.

        Returns:
            The computed contrastive loss. A scalar.
        """
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
        """
        Compute the cosine loss between image features and combined features.

        Args:
            image_features: The image features. A tensor of shape (batch_size, feature_dim).
            combined_features: The combined features. A tensor of shape (batch_size, feature_dim).
            device: The device to use for computation. Defaults to the device set by geoopt.

        Returns:
            The computed cosine loss. A scalar.
        """
        target = torch.ones(image_features.size(0)).to(device)
        output = self.loss(image_features, combined_features, target)

        return output


class LabelContrastiveLoss(
    nn.Module
):  # BUG: This is not work as expected, label embeddings should work in a different way, see notes
    def __init__(
        self,
        margin: float = 0.2,
        lambda_pos: float = 1.0,
        lambda_neg: float = 1.0,
        lambda_labelchange: float = 0.1,
        lambda_preserve: float = 0.1,
        lambda_angular: float = 0.5,
        lambda_pull_away: float = 0.5,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        print("Using Combined Cosine and Contrastive Loss")
        self.margin = margin
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.lambda_labelchange = lambda_labelchange
        self.lambda_preserve = lambda_preserve
        self.lambda_angular = lambda_angular
        self.lambda_pull_away = lambda_pull_away
        self.return_dict = return_dict
        # TODO Add diversity loss to encourage more diversity in the embeddings

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        combined_features: Tensor,
        combined_features_neg: Tensor,
        label_embedding: Tensor,
        label_embedding_proj: Tensor,
    ):
        # Compute cosine similarity
        cos_pos = F.cosine_similarity(
            combined_features, image_features, dim=-1
        )  # Positive contrast
        cos_orig = F.cosine_similarity(text_features, image_features, dim=-1)  # Original contrast
        cos_neg = F.cosine_similarity(
            combined_features_neg, image_features, dim=-1
        )  # Negative cvidiaontrast

        loss_improve = torch.clamp(
            cos_orig + self.margin - cos_pos, min=0
        ).mean()  # Let combined features be closer to image features
        loss_neg = torch.clamp(
            cos_pos - cos_neg + self.margin, min=0
        ).mean()  # Let combined features be further from neg

        label_change_loss = label_change_regularizer(
            combined_features, text_features, label_embedding_proj, alpha=self.lambda_labelchange
        )

        text_preserve_loss = text_preserve_regularizer(
            combined_features, text_features, alpha=self.lambda_preserve
        )

        boundary_loss = boundary_penalty(label_embedding, radius=3.0, alpha=0.1)

        angular_loss = angular_consistency_loss(
            label_embedding_proj, text_features, combined_features, alpha=self.lambda_angular
        )

        pull_away_loss = pull_away_diversity_loss(
            label_embedding_proj, alpha=self.lambda_pull_away
        )

        total_loss = (
            self.lambda_pos * loss_improve
            + self.lambda_neg * loss_neg
            + angular_loss
            + pull_away_loss
            + self.lambda_labelchange * label_change_loss
            + self.lambda_preserve * text_preserve_loss
            + boundary_loss
        )

        # Before first k epoch, we not contrast, but rather fully improve over original
        early_loss = loss_improve + boundary_loss  # + 0.5 * loss_neg

        loss_dict = {
            "loss_improve": loss_improve,
            "loss_neg": loss_neg,
            "loss_label_change": label_change_loss,
            "loss_preserve": text_preserve_loss,
            "loss_angular": angular_loss,
            "loss_pull_away": pull_away_loss,
            "loss_boundary": boundary_loss,
            "total_loss": total_loss,
            "early_loss": early_loss,
        }

        if self.return_dict:
            return loss_dict
        else:
            return total_loss


# TODO: Implement a classifier that choose labels during training, for example, to do concatenation of image, text and label embeddings.


def main(): ...


if __name__ == "__main__":
    main()


# # Positive contrast (diagonal should be similar)
# positive_similarity = self.cosine_similarity(image_features, combined_features)
# positive_loss = self.cosine_loss(image_features, combined_features, target_positive)

# # Negative contrast (diagonal should be dissimilar)
# target_negative = -torch.ones(batch_size).to(device)
# negative_similarity = self.cosine_similarity(image_features, combined_features_neg)
# negative_loss = self.cosine_loss(image_features, combined_features_neg, target_negative)

# # Combine both positive and negative loss for contrastive learning
# label_contrastive_loss = positive_loss + negative_loss


# 2. Contrastive loss with positive and negative contrast if combined_features_neg is provided
# label_contrastive_loss = 0
# if combined_features_neg is not None and self.label_weight > 0:
#     # Part 1: let (txt1 - txt2) = D1, (Comb1-Comb2) = D2, then (D1 - D2) / 2 = D3.
#     # Assume txt1 and txt2 is combined with lab1 and lab2 respectively.
#     # Then, if D3 is higher, it means that lab1 and lab2 are dissimilar.
#     # If D3 is lower, it means that lab1 and lab2 are similar.
#     # We define the metric to measure the difference to be cosine similarity between txt1 and txt2.

#     D1 = compute_cosine_similarity(text_features, text_features)
#     D2 = compute_cosine_similarity(combined_features, combined_features)

#     D3 = (D1 - D2) / 2

#     D3_loss = self.cosine_loss(
#         text_features, combined_features, D3
#     )  # TODO: Check if this is correct

#     # Part 2: let (Comb1 - txt1) = D3, (Comb2 - txt2) = D4, then (D3 + D4) / 2 = D5.
#     # D5 works as a regularizer, such that the combined features should not be too far from the text features.

# # 3. Contrastive loss with the difference of combined_features and raw_features
# diff_contrastive_loss = 0
# if text_features is not None and self.diff_weight > 0:
#     raw_sim = compute_cosine_similarity(image_features, text_features)
#     comb_sim = compute_cosine_similarity(image_features, combined_features)

#     diff_sim = comb_sim - raw_sim

#     if diff_sim.dim() != 2:
#         raise ValueError(
#             "s loss: diagonal elements (positive pairs), we want sim_diff to be greater than margin_pos
#     pos_loss = torch.relu(self.margin_pos - diff_sim.diag()).mean()

#     # Negative loss: off-diagonal elements (negative pairs), we want sim_diff to be less than margin_neg
#     # We mask out the diagonal elements using torch.eye(batch_size)
#     mask = torch.eye(batch_size, dtype=torch.bool)
#     neg_loss = torch.relu(diff_sim[~mask] - self.margin_neg).mean()
#     diff_contrastive_loss = pos_loss + neg_lossim_diff should be a 2D matrix, but got a tensor with shape {}".format(
#                 diff_sim.shape
#             )
#         )
#     # Positive
