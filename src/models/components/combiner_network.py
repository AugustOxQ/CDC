# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Sagar Vaze from https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# Code from: https://github.com/ABaldrati/CLIP4CirDemo/blob/main/model.py
# and https://raw.githubusercontent.com/facebookresearch/genecis/main/models/combiner_model.py


from collections import deque
from cProfile import label

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.components.simple_attention import (
    SimpleResidule,
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


class Combiner_basic_low(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()
        self.text_projection_layer = SimpleResidule(
            input_dim=clip_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout_rate=0.5,
            residual=True,
        )
        self.label_projection_layer = SimpleResidule(
            input_dim=label_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            dropout_rate=0.5,
            residual=False,
        )

        self.combiner_layer = SimpleResidule(
            input_dim=projection_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=clip_feature_dim,
            dropout_rate=0.5,
            residual=False,
        )

        self.output_layer = nn.Linear(clip_feature_dim, clip_feature_dim)

        self.dropout = nn.Dropout(0.5)

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
        :param label_features: Label features (shape: batch, label_dim)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        text_projected_features = self.text_projection_layer(text_features)

        label_projected_features = self.label_projection_layer(label_features)

        raw_combined_features = torch.cat((text_projected_features, label_projected_features), -1)

        combined_features = self.combiner_layer(raw_combined_features)

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


class Combiner_add(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        self.combiner_layer = SimpleResidule(
            input_dim=projection_dim + label_dim,
            hidden_dim=hidden_dim,
            output_dim=clip_feature_dim,
            dropout_rate=0.5,
            residual=False,
        )

        self.output_layer = nn.Linear(clip_feature_dim, clip_feature_dim)

        self.dropout = nn.Dropout(0.5)

        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim + label_dim, hidden_dim),
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
        :param label_features: Label features (shape: batch, label_dim)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        raw_combined_features = torch.cat((text_features, label_features), -1)

        combined_features = self.combiner_layer(raw_combined_features)

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        # print(dynamic_scalar.shape) # (batch, 1)
        self.scalar.add(dynamic_scalar.mean().item())
        # print(self.scalar.get())

        # # Option1: Output is a combination of combined_featured and text_features and label_projected_features
        output = (
            dynamic_scalar * self.output_layer(combined_features)
            + (1 - dynamic_scalar) * text_features
        )

        # Option2: Output is a combination of combined_featured and text_features
        # output = (
        #     dynamic_scalar * self.output_layer(combined_features)
        #     + (1 - dynamic_scalar) * text_features
        # )

        # Option3: Output is combined_features
        # output = self.output_layer(combined_features) + text_features

        return F.normalize(output)


class Combiner_add_multi(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
        warm_up_epoch: int = 5,
        scale_init: float = 100,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        self.label_proj_layer = nn.Linear(label_dim, projection_dim)
        nn.init.orthogonal_(self.label_proj_layer.weight)  # Orthogonal initialization

        self.output_layer = nn.Linear(clip_feature_dim, clip_feature_dim)
        self.warm_up_epoch = warm_up_epoch

        self.dropout = nn.Dropout(0.5)

        self.scale = nn.Parameter(torch.ones(1) * scale_init)

        # Larger dynamic scalar means more weight on the combined features
        self.scalar = FixedSizeQueue(10)

    def print_scalar(self):

        return self.scalar.get()

    def get_newest(self):
        return self.scalar.get_newest()

    @torch.jit.export
    def forward(
        self, text_features: Tensor, text_full: Tensor, label_features: Tensor, epoch: None
    ) -> Tensor:
        """Combine the text features and label features using attention.

        Outputs combined features.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, label_dim)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"
        label_proj = self.label_proj_layer(label_features)
        output = text_features + 100 * label_proj  # Or self.scale

        self.scalar.add(self.scale.item())

        return F.normalize(output)


class Combiner_add_attention(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
        warm_up_epoch: int = 5,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        self.label_proj = nn.Sequential(
            SimpleResidule(
                input_dim=label_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout_rate=0.5,
                residual=False,
            ),
            SimpleResidule(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout_rate=0.5,
                residual=True,
            ),
            SimpleResidule(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout_rate=0.5,
                residual=True,
            ),
            SimpleResidule(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=clip_feature_dim,
                dropout_rate=0.5,
                residual=False,
            ),
        )

        # Define Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=clip_feature_dim,  # feature dimension
            nhead=num_heads,
            dim_feedforward=hidden_dim,  # feedforward hidden size
            batch_first=True,  # (batch, seq, dim)
            dropout=0.5,
        )

        # Stack multiple layers
        self.combiner_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(clip_feature_dim, clip_feature_dim)
        self.warm_up_epoch = warm_up_epoch

        self.dropout = nn.Dropout(0.5)

        self.dynamic_scalar = nn.Sequential(
            nn.Linear(clip_feature_dim, hidden_dim),
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
    def forward(
        self, text_features: Tensor, text_full: Tensor, label_features: Tensor, epoch: None
    ) -> Tensor:
        """Combine the text features and label features using attention.

        Outputs combined features.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, label_dim)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        # label_proj = F.pad(label_features, (0, text_features.shape[1] - label_features.shape[1]), mode='constant', value=0)
        label_proj = self.label_proj(label_features)

        enriched_global = text_features + label_proj
        enriched_global_expanded = enriched_global.unsqueeze(1)
        fused_input = torch.cat([enriched_global_expanded, text_full], dim=1)

        combined_features = self.combiner_layer(fused_input)
        # print(combined_features.shape)

        dynamic_scalar = self.dynamic_scalar(combined_features).mean(dim=1)
        self.scalar.add(dynamic_scalar.mean().item())

        output = (
            self.output_layer(combined_features.mean(dim=1))
            + dynamic_scalar * text_features
            + (1 - dynamic_scalar) * label_proj
        )

        return F.normalize(output)


class Combiner_add_multi2(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
        warm_up_epoch: int = 5,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()
        modules = [
            SimpleResidule(
                input_dim=projection_dim + label_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout_rate=0.5,
                residual=False,
            ),
        ]
        for _ in range(num_layers - 2):
            modules.append(
                SimpleResidule(
                    input_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout_rate=0.5,
                    residual=False,
                )
            )

        modules.append(
            SimpleResidule(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=clip_feature_dim,
                dropout_rate=0.5,
                residual=False,
            )
        )

        self.combiner_layer = nn.Sequential(*modules)

        self.output_layer = nn.Linear(clip_feature_dim, clip_feature_dim)
        self.warm_up_epoch = warm_up_epoch

        self.dropout = nn.Dropout(0.5)

        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim + label_dim, hidden_dim),
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
    def forward(
        self, text_features: Tensor, text_full: Tensor, label_features: Tensor, epoch: None
    ) -> Tensor:
        """Combine the text features and label features using attention.

        Outputs combined features.
        :param text_features: CLIP textual features (shape: batch, 512)
        :param text_full: CLIP textual features with full sequence length (shape: batch, L, 512)
        :param label_features: Label features (shape: batch, label_dim)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        raw_combined_features = torch.cat((text_features, label_features), -1)

        combined_features = self.combiner_layer(raw_combined_features)

        if epoch is not None:
            dynamic_scalar = min(1.0, max(0.5, epoch / self.warm_up_epoch))
        else:
            dynamic_scalar = 1.0
        self.scalar.add(dynamic_scalar)
        # print(dynamic_scalar.shape) # (batch, 1)
        # print(self.scalar.get())

        # # Option1: Output is a combination of combined_featured and text_features and label_projected_features
        output = (
            dynamic_scalar * self.output_layer(combined_features)
            + (1 - dynamic_scalar) * text_features
        )

        return F.normalize(output)


class Combiner_add2(nn.Module):
    """Combiner module which once trained fuses textual and label information."""

    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
    ) -> None:
        """
        :param clip_feature_dim: CLIP input feature dimension (e.g., 512)
        :param projection_dim: projection dimension (e.g., 256)
        :param hidden_dim: hidden dimension (e.g., 512)
        :param num_heads: Number of heads in multi-head attention
        :param num_layers: Number of transformer layers
        """
        super().__init__()

        proj_matrix = torch.randn(label_dim, clip_feature_dim, device="cuda")
        self.proj_matrix = proj_matrix / proj_matrix.norm(dim=0, keepdim=True)

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
        :param label_features: Label features (shape: batch, label_dim)
        :return: combined textual features (shape: batch, 512)
        """
        assert (
            len(text_full.shape) == 3
        ), f"text_full should be of shape (batch, L, 512), instead get {text_full.shape}"

        if label_features.shape[1] != text_features.shape[1]:
            # padding label_features to match text_features
            label_features = label_features @ self.proj_matrix
        output = text_features + label_features

        return F.normalize(output)


class Combiner_cross_attention(nn.Module):
    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
    ) -> None:
        super().__init__()
        self.label_proj = nn.Linear(label_dim, clip_feature_dim)  # in case label_dim != dim

        self.cls_attn = nn.MultiheadAttention(
            embed_dim=clip_feature_dim, num_heads=num_heads, batch_first=True
        )
        self.seq_attn = nn.MultiheadAttention(
            embed_dim=clip_feature_dim, num_heads=num_heads, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)
        self.scalar = FixedSizeQueue(10)

    def forward(self, text_features, text_full, label_features):
        # Ensure label features are shaped (B, 1, D)
        label_feat = self.label_proj(label_features).unsqueeze(1)

        # Text CLS attends to label
        text_cls_attn, _ = self.cls_attn(
            query=text_features.unsqueeze(1), key=label_feat, value=label_feat
        )  # (B, 1, D)

        # Full sequence attends to label
        text_seq_attn, _ = self.seq_attn(
            query=text_full, key=label_feat, value=label_feat
        )  # (B, L, D)

        # Aggregate token-level attention (e.g., mean)
        text_seq_pooled = text_seq_attn.mean(dim=1)  # (B, D)

        # Combine CLS and token-level attended results
        fused = torch.cat([text_cls_attn.squeeze(1), text_seq_pooled], dim=-1)
        fused = self.ffn(fused)

        return F.normalize(self.output_layer(fused), dim=-1)


class Combiner_transformer(nn.Module):
    def __init__(
        self,
        clip_feature_dim: int = 512,
        projection_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        label_dim: int = 512,
    ) -> None:
        super().__init__()
        self.label_proj = nn.Linear(label_dim, clip_feature_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=clip_feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)
        self.scalar = FixedSizeQueue(10)

    def forward(self, text_features, text_full, label_features):
        B = text_features.size(0)
        label_feat = self.label_proj(label_features).unsqueeze(1)  # (B, 1, D)

        # Build a sequence: [text_cls] + [text_tokens] + [label_feat]
        text_cls = text_features.unsqueeze(1)  # (B, 1, D)
        sequence = torch.cat([text_cls, text_full, label_feat], dim=1)  # (B, L+2, D)

        encoded = self.encoder(sequence)  # (B, L+2, D)

        # Output the updated CLS token
        return F.normalize(self.output_layer(encoded[:, 0, :]), dim=-1)


def test_forward_variables_shape_and_type():
    combiner = Combiner_add_attention(
        clip_feature_dim=512,
        projection_dim=512,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        label_dim=32,
    )
    text_features = torch.randn(2, 512)
    text_full = torch.randn(2, 77, 512)

    for i in range(20):
        label_features = torch.randn(2, 32)
        output = combiner.forward(text_features, text_full, label_features, 0)
        print(output.shape, f"{output[0][:10].tolist()}")


def main():
    test_forward_variables_shape_and_type()


if __name__ == "__main__":
    main()
