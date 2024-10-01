from email.mime import image
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
        loss = 0.5 * (
            (1 - label) * distances.pow(2) + label * F.relu(self.margin - distances).pow(2)
        )
        return loss.mean()


def cross_entropy(preds, targets, reduction="none") -> torch.Tensor:
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class ClipLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        print("Using ClipLoss")
        self.temperature = torch.ones([]) * (1 / 0.07)

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        device: torch.device = device,
    ):
        image_features, text_features = norm_features(image_features, text_features)

        logits = image_features @ text_features.T * self.temperature
        sim = image_features @ text_features.T
        targets = F.softmax(sim + sim.t() / 2 * self.temperature, dim=-1)
        image_loss = cross_entropy(logits, targets, reduction="none")
        text_loss = cross_entropy(logits.t(), targets.t(), reduction="none")
        loss = (image_loss + text_loss) / 2.0

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
        combined_features: Tensor,
        device: torch.device = device,
    ):
        target = torch.ones(image_features.size(0)).to(device)
        output = self.loss(image_features, combined_features, target)

        return output


class LabelContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        print("Using Combined Cosine and Contrastive Loss")
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)
        # TODO Add diversity loss to encourage more diversity in the embeddings

    def forward(
        self,
        image_features: Tensor,
        combined_features: Tensor,
        combined_features_neg: Tensor,
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
        total_loss = cosine_loss + contrastive_loss

        return total_loss


class CosineLossR(nn.Module):
    def __init__(
        self, margin: float = 0.1, reg_threshold: float = 0.3, reg_weight: float = 0.1
    ) -> None:
        super().__init__()
        print("Using CosineLoss with regularizer")
        self.margin = margin
        self.loss = nn.CosineEmbeddingLoss(margin=margin)
        self.reg_threshold = reg_threshold  # Cosine similarity threshold for regularizer
        self.reg_weight = reg_weight  # Weight for regularizer loss

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        comb_features: Tensor,
        device: torch.device,
    ):
        # Step 1: Cosine loss between image_features and comb_features (improved text features)
        target = torch.ones(image_features.size(0)).to(device)
        cosine_loss = self.loss(image_features, comb_features, target)

        # Step 2: Regularizer based on similarity between image_features and original text_features
        cosine_sim = F.cosine_similarity(image_features, text_features, dim=-1)

        # If cosine similarity is higher than threshold, the regularizer is low; otherwise, it's high
        regularizer = torch.where(
            cosine_sim > self.reg_threshold,
            torch.zeros_like(cosine_sim),  # No penalty if similarity is high
            self.reg_weight * (1 - cosine_sim),  # Penalize if similarity is low
        )

        # Step 3: Combine the cosine loss and regularizer loss
        total_loss = cosine_loss + regularizer.mean()  # Mean regularizer loss

        return total_loss


class DiversityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        print("Using DiversityLoss")
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        device: torch.device = device,
    ):
        # image_features, text_features = norm_features(image_features, text_features)
        output = self.loss(image_features, text_features)
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
