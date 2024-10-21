import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrossAttention(nn.Module):
    """A simple implementation of cross attention module."""

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.query_projection = nn.Linear(embed_dim, embed_dim)  # For projecting the image feature
        self.key_projection = nn.Linear(embed_dim, embed_dim)  # For projecting the text features
        self.value_projection = nn.Linear(embed_dim, embed_dim)  # For projecting the text features
        self.output_projection = nn.Linear(
            embed_dim, embed_dim
        )  # Final projection after attention

    def forward(self, cls_feature: Tensor, full_feature: Tensor) -> Tensor:
        """
        Args:
            cls_feature: Tensor of shape [batch, 512] (global feature)
            full_feature: Tensor of shape [batch, 77, 512] (overall feature)

        Returns:
            Fused cls feature of shape [batch, 512]
        """
        # [batch, 512] -> [batch, 1, 512]
        cls_feature = cls_feature.unsqueeze(1)  # Add an extra dimension to make it [batch, 1, 512]

        # Linear projections
        query = self.query_projection(cls_feature)  # [batch, 1, 512]
        key = self.key_projection(full_feature)  # [batch, 77, 512]
        value = self.value_projection(full_feature)  # [batch, 77, 512]

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # [batch, 1, 77]
        attention_scores = attention_scores / torch.sqrt(
            torch.tensor(key.size(-1), dtype=torch.float32)
        )

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, 1, 77]

        # Compute the weighted sum of the values
        attended_value = torch.matmul(attention_weights, value)  # [batch, 1, 512]

        # Squeeze out the extra dimension -> [batch, 512]
        attended_value = attended_value.squeeze(1)

        # Project the attended value back to original dimension
        output = self.output_projection(attended_value)  # [batch, 512]

        return output


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        """Initialize a SimpleTransformer instance.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads in the Transformer layer.
            hidden_dim (int): The dimensionality of the feed-forward layer in the
                Transformer layer.
            num_layers (int): The number of Transformer layers to use.
        """
        super().__init__()

        # Transformer Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=0.5
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear projection to reduce the dimension back to [batch, 512]
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, cls_feature, full_feature):
        """
        Args:
            cls_feature: Tensor of shape [batch, 512] (global feature)
            full_feature: Tensor of shape [batch, 77, 512] (overall feature)

        Returns:
            Fused feature of shape [batch, 77, 512]
        """
        # [batch, 512] -> [batch, 1, 512]
        cls_feature = cls_feature.unsqueeze(1)  # Add an extra dimension to make it [batch, 1, 512]

        # Concatenate cls_feature and full_feature to form a sequence [batch, 78, 512]
        combined_features = torch.cat((cls_feature, full_feature), dim=1)

        # Apply the Transformer encoder
        transformer_output = self.transformer_encoder(combined_features)  # [batch, 78, 512]

        # avg pooling to get the fused feature [batch, 512]
        updated_image_feature = torch.mean(transformer_output, dim=1)

        # Project the result back to the original embedding dimension
        fused_image_feature = self.output_projection(updated_image_feature)  # [batch, 512]

        return fused_image_feature


class SimpleTransformer2(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        """Initialize a SimpleTransformer instance.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads in the Transformer layer.
            hidden_dim (int): The dimensionality of the feed-forward layer in the
                Transformer layer.
            num_layers (int): The number of Transformer layers to use.
        """
        super().__init__()

        # Transformer Encoder layer
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=0.5,
            batch_first=True,
        )

        # Linear projection to reduce the dimension back to [batch, 512]
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, cls_feature, full_feature):
        """
        Args:
            cls_feature: Tensor of shape [batch, 512] (global feature)
            full_feature: Tensor of shape [batch, 77, 512] (overall feature)

        Returns:
            Fused feature of shape [batch, 77, 512]
        """
        # [batch, 512] -> [batch, 1, 512]
        cls_feature = cls_feature.unsqueeze(1)  # Add an extra dimension to make it [batch, 1, 512]

        # Apply the Transformer, src = cls_feature, tgt = full_feature, the output shape is [batch, 77, 512]
        transformer_output = self.transformer(cls_feature, full_feature)

        # avg pooling to get the fused feature [batch, 512]
        updated_image_feature = torch.mean(transformer_output, dim=1)

        # Project the result back to the original embedding dimension
        fused_image_feature = self.output_projection(updated_image_feature)  # [batch, 512]

        return fused_image_feature


class SimpleResidule(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=512,
        dropout_rate=0.5,
        output_dim=512,
        residual=True,
    ) -> None:
        super().__init__()

        # First fully connected layer (input -> hidden)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second fully connected layer (hidden -> input) with residual connection
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Activation function
        self.activation = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove the second dimension to shape [batch, 512]
        # First layer with batch normalization, activation, and dropout
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)

        # Second layer with batch normalization, activation, and dropout
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        if self.residual:
            # Add the residual connection (input + transformed output)
            out = out + x

        return self.activation(out)  # Apply activation to the output with the residual connection


def test_simple_attention():
    model = SimpleResidule(32, 512, 0.5, 512, False)
    input = torch.randn(64, 32)
    output = model(input)
    print(output.shape)


def main():
    test_simple_attention()


if __name__ == "__main__":
    main()
