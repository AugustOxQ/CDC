import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class LowCrossAttention(nn.Module):
    def __init__(self, D_text=512, D_label=128, D_hidden=512, num_heads=8):
        super().__init__()
        # Project text to label dim
        self.query_down = nn.Linear(D_text, D_label)
        self.attn = nn.MultiheadAttention(embed_dim=D_label, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(D_label, D_text)  # Back to original dim

    def forward(self, text_cls, label_feats):
        """
        text_cls: (B, 1, D_text)
        label_feats: (B, N, D_label)
        """
        q = self.query_down(text_cls)  # (B, 1, D_label)
        attn_out, _ = self.attn(q, label_feats, label_feats)
        return self.out_proj(attn_out)  # (B, 1, D_text)


class Combiner_cross_attention_to_label_lowD(nn.Module):
    def __init__(self, D_text=512, D_label=128, hidden_dim=512, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(D_text, D_label)
        self.attn = nn.MultiheadAttention(embed_dim=D_label, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Linear(D_label, D_text)

        self.ffn = nn.Sequential(
            nn.Linear(D_text * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, D_text),
        )

    def forward(self, text_features: Tensor, text_full: Tensor, label_features: Tensor) -> Tensor:
        """
        text_features: (B, D_text)
        text_full: (B, L, D_text)
        label_features: (B, N, D_label)
        """
        B = text_features.shape[0]

        # CLS attends to label features
        query_cls = self.query_proj(text_features.unsqueeze(1))  # (B, 1, D_label)
        attn_cls, _ = self.attn(query_cls, label_features, label_features)  # (B, 1, D_label)
        attn_cls = self.output_proj(attn_cls.squeeze(1))  # (B, D_text)

        # Full sequence attends to label features
        query_full = self.query_proj(text_full)  # (B, L, D_label)
        attn_full, _ = self.attn(query_full, label_features, label_features)  # (B, L, D_label)
        attn_full = self.output_proj(attn_full)  # (B, L, D_text)
        attn_full_pooled = attn_full.mean(dim=1)  # (B, D_text)

        # Fuse
        fused = torch.cat([attn_cls, attn_full_pooled], dim=-1)  # (B, 2*D_text)
        fused = self.ffn(fused)
        return F.normalize(fused, dim=-1)


def test_simple_attention():
    model = SimpleResidule(32, 512, 0.5, 512, False)
    input = torch.randn(64, 32)
    output = model(input)
    print(output.shape)


def main():
    test_simple_attention()


if __name__ == "__main__":
    main()
