# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# Code from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# and https://raw.githubusercontent.com/facebookresearch/genecis/main/models/combiner_model.py


from collections import deque

import torch
import torch.nn.functional as F
from torch import nn


class FixedSizeQueue:
    def __init__(self, max_size: int):
        self.queue = deque(maxlen=max_size)

    def add(self, number: float):
        self.queue.append(number)

    def get(self):
        # Return a list of floats formatted to 2 decimal places
        return [f"{num:.2f}" for num in self.queue]

    def get_newest(self):
        # Return the most recent element formatted to 2 decimal places
        if self.queue:
            return self.queue[-1]  # The last element in the deque
        else:
            return None  # In case the queue is empty


class Combiner_basic(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param clip_feature_dim: CLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
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

        self.logit_scale = 100

    @torch.jit.export
    def forward(self, text_features, label_features):
        """Cobmine the text features and label features.

        It outputs the predicted features
        :param text_features: CLIP textual features (shape: batch, 512)
        :param label_features: Label features (shape: batch, 512)
        :return: predicted textual features (shape: batch, 512)
        """

        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        label_projected_features = self.dropout2(
            F.relu(self.image_projection_layer(label_features))
        )

        raw_combined_features = torch.cat((text_projected_features, label_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * label_projected_features
        )

        return F.normalize(output)


class Combiner(nn.Module):
    """Combiner module using cross-attention for combining textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ):
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        # Projection layers
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.label_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        # Transformer encoder for cross-attention
        self.cross_attention = nn.Transformer(
            d_model=projection_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=0.5,
        )

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # Combiner layer and output layer
        self.combiner_layer = nn.Linear(projection_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

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
    def forward(self, text_features, text_full, label_features):
        """Combine the text features and label features using attention.

        Outputs predicted features.
        :param text_features: CLIP textual features (shape: batch, 77, 512)
        :param label_features: Label features (shape: batch, 512)
        :return: predicted textual features (shape: batch, 512)
        """

        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        # Project text features (batch, 77, 512) -> (batch, 77, projection_dim)
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_full)))

        # Project label features (batch, 512) -> (batch, projection_dim), unsqueeze for broadcasting
        label_projected_features = self.dropout2(
            F.relu(self.label_projection_layer(label_features))
        )
        label_projected_features = label_projected_features.unsqueeze(1).repeat(
            1, text_projected_features.size(1), 1
        )

        # Apply cross-attention (batch, 77, projection_dim)
        # Transformer expects input in (seq_len, batch, feature_dim)
        combined_features = self.cross_attention(
            text_projected_features.permute(1, 0, 2),  # text as query
            label_projected_features.permute(1, 0, 2),  # label as key/value
        ).permute(1, 0, 2)
        # print(combined_features.shape) # (batch, 77, projection_dim)

        # Apply combiner and output layer
        combined_features = self.combiner_layer(
            combined_features.mean(dim=1)
        )  # Pool over sequence dimension
        # print(combined_features.shape) # (batch, projection_dim)

        # Dynamic scalar
        dynamic_scalar = self.dynamic_scalar(combined_features)
        # print(dynamic_scalar.shape) # (batch, 1)
        self.scalar.add(dynamic_scalar.mean().item())
        # print(self.scalar.get())

        # print(combined_features.shape) # (batch, projection_dim)

        # Option1: Output is a combination of all three inputs
        output = (
            self.output_layer(combined_features)
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * label_features
        )

        # Option2: Output is a combination of combined_featured and text_features
        # output = dynamic_scalar * self.output_layer(combined_features) + (1 - dynamic_scalar) * text_features

        # Option3: Output is combined_features
        # output = self.output_layer(combined_features)

        return F.normalize(output)


def test_forward_variables_shape_and_type():
    combiner = Combiner()
    text_features = torch.randn(2, 512)
    text_full = torch.randn(2, 77, 512)
    label_features = torch.randn(2, 512)

    for i in range(20):
        output = combiner.forward(text_features, text_full, label_features)
        print(type(combiner.get_newest()))


def main():
    test_forward_variables_shape_and_type()


if __name__ == "__main__":
    main()
