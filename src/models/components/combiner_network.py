# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# Code from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# and https://raw.githubusercontent.com/facebookresearch/genecis/main/models/combiner_model.py


from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.simple_attention import (
    CrossAttention,
    SimpleResidule,
    SimpleTransformer,
    SimpleTransformer2,
)


class FixedSizeQueue:
    def __init__(self, max_size: int) -> None:
        """Initialize a FixedSizeQueue with a given maximum size.

        Args:
            max_size: The maximum size of the queue.

        The queue is implemented as a deque with a maximum length equal to max_size.
        When the queue is full and another element is added, the oldest element is removed.
        """
        self.queue = deque(maxlen=max_size)

    def add(self, number: float) -> None:
        """Add a number to the end of the queue. If the queue is full, remove the oldest element
        first.

        Args:
            number: The number to add to the queue.
        """
        self.queue.append(number)

    def get(self) -> list[str]:
        """Return a list of the current elements in the queue as strings, each formatted to 2
        decimal places.

        Returns:
            A list of strings, where each string is a number from the queue formatted to 2 decimal places.
        """
        return [f"{num:.2f}" for num in self.queue]

    def get_newest(self) -> float:
        """Return the newest element in the queue.

        If the queue is not empty, return the last element in the deque.
        If the queue is empty, return -1.0.

        Returns:
            The newest element in the queue, or -1.0 if the queue is empty.
        """
        if self.queue:
            return self.queue[-1]  # The last element in the deque
        else:
            return -1.0  # In case the queue is empty


class Combiner_basic(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Larger dynamic scalar means more weight on the combined features
        self.scalar = FixedSizeQueue(10)

    def print_scalar(self):
        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    @torch.jit.export
    def forward(self, text_features: Tensor, text_full: Tensor, label_features: Tensor) -> Tensor:
        """Combine the text features and label features using attention.

        Outputs combined features.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, 512)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        label_projected_features = self.dropout2(
            F.relu(self.image_projection_layer(label_features))
        )

        raw_combined_features = torch.cat((text_projected_features, label_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        # print(dynamic_scalar.shape) # (batch, 1)
        self.scalar.add(dynamic_scalar.mean().item())
        # print(self.scalar.get())

        # # Option1: Output is a combination of combined_featured and text_features and label_projected_features
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * label_projected_features
        )

        # Option2: Output is a combination of combined_featured and text_features
        # output = (
        #     dynamic_scalar * self.output_layer(combined_features)
        #     + (1 - dynamic_scalar) * text_features
        # )

        # Option3: Output is combined_features
        # output = self.output_layer(combined_features) + text_features

        return F.normalize(output)


class Combiner_cross_attention(nn.Module):
    """Combiner module using transformer for combining textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 512)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(projection_dim)

        # Multi-head attention for label attending to text
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=projection_dim, num_heads=num_heads, batch_first=True
        )

        # Additional scalar dynamic weighting
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.label_dropout = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.9),
            nn.Linear(projection_dim, projection_dim),
        )

        self.label_encoder = SimpleResidule(input_dim=projection_dim, dropout_rate=0.5)

        self.output_layer = nn.Linear(projection_dim, clip_feature_dim)

        # Larger dynamic scalar means more weight on the combined features
        self.scalar = FixedSizeQueue(10)

    def print_scalar(self):
        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    def forward(self, text_features: Tensor, text_full: Tensor, label_features: Tensor) -> Tensor:
        """Combine the text features and label features using cross-attention.

        Outputs combined features with the shape of text_full.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, 512)
        :return: combined textual features (shape: batch, L, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead got {text_full.shape}"

        # Reshape label_features to (batch, 1, projection_dim) to act as both query and value

        label_features = self.label_encoder(label_features)

        label_features = label_features.unsqueeze(1)  # shape: (batch, 1, projection_dim)

        # Cross-attention: text_full attends to label_features
        attended_label_features, _ = self.cross_attention(
            query=text_full,  # shape: (batch, L, projection_dim) -> Query
            key=label_features,  # shape: (batch, 1, projection_dim) -> Key
            value=label_features,  # shape: (batch, 1, projection_dim) -> Value
        )  # Output shape: (batch, L, projection_dim)

        # Mean pool the attended label features over dim 1
        attended_label_features = torch.mean(
            attended_label_features, dim=1
        )  # (batch, projection_dim)

        # Dynamic scalar
        dynamic_scalar = self.dynamic_scalar(attended_label_features)
        self.scalar.add(dynamic_scalar.mean().item())

        # Label Dropout
        label_features_dropout = self.label_dropout(label_features)

        # Skip-connection and normalization
        output = self.batch_norm(
            attended_label_features
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * label_features_dropout.squeeze(1)
        )

        return output  # Return the output


# Hook for TorchScript
class Combiner_transformer(nn.Module):
    """Combiner module using transformer for combining textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        self.simple_transformer = SimpleTransformer(
            embed_dim=projection_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.batch_norm = nn.BatchNorm1d(projection_dim)

        # Additional scalar dynamic weighting
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Larger dynamic scalar means more weight on the combined features
        self.scalar = FixedSizeQueue(10)

    def print_scalar(self):
        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    @torch.jit.export
    def forward(self, text_features: Tensor, text_full: Tensor, label_features: Tensor) -> Tensor:
        """Combine the text features and label features using cross-attention.

        Outputs combined features.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, 512)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        combined_features = self.simple_transformer(label_features, text_full)

        # Dynamic scalar
        dynamic_scalar = self.dynamic_scalar(combined_features)
        self.scalar.add(dynamic_scalar.mean().item())

        # Skip-connection and normalization
        output = self.batch_norm(
            dynamic_scalar * combined_features + (1 - dynamic_scalar) * text_features
        )

        return output


class Combiner_transformer2(nn.Module):
    """Combiner module using transformer for combining textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        self.simple_transformer = SimpleTransformer2(
            embed_dim=projection_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Additional scalar dynamic weighting
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.batch_norm = nn.BatchNorm1d(projection_dim)

        # Larger dynamic scalar means more weight on the combined features
        self.scalar = FixedSizeQueue(10)

    def print_scalar(self):
        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    @torch.jit.export
    def forward(self, text_features: Tensor, text_full: Tensor, label_features: Tensor) -> Tensor:
        """Combine the text features and label features using cross-attention.

        Outputs combined features.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, 512)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        combined_features = self.simple_transformer(label_features, text_full)

        # Dynamic scalar
        dynamic_scalar = self.dynamic_scalar(combined_features)
        self.scalar.add(dynamic_scalar.mean().item())

        # Skip-connection and normalization
        output = self.batch_norm(
            combined_features
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * label_features
        )

        return output


def test_forward_variables_shape_and_type():
    combiner = Combiner_cross_attention()
    text_features = torch.randn(2, 512)
    text_full = torch.randn(2, 77, 512)
    label_features = torch.randn(2, 512)

    for i in range(20):
        output = combiner.forward(text_features, text_full, label_features)
        print(output.shape)


def main():
    test_forward_variables_shape_and_type()


if __name__ == "__main__":
    main()
