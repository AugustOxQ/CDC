import torch
import torch.nn.functional as F


def boundary_penalty(embeddings, radius=1.0, alpha=0.1):
    norms = torch.norm(embeddings, p=2, dim=1)
    penalty = torch.where(norms > radius, (norms - radius) ** 2, torch.zeros_like(norms))
    return alpha * torch.mean(penalty)


def l2_regularizer(embeddings, alpha=0.1):
    l2_norm = torch.norm(embeddings, p=2, dim=1)  # Compute L2 norm for each embedding
    return alpha * torch.mean(l2_norm**2)  # Return the mean L2 norm with scaling


def text_preserve_regularizer(text_features, combined_features, tau=0.2, alpha=0.1):
    delta = (combined_features - text_features).norm(dim=-1)  # [B]
    excess_change = F.relu(delta - tau)
    return alpha * excess_change.pow(2).mean()


def label_change_regularizer(text_features, combined_features, label_features, tau=0.2, alpha=0.1):
    delta = (combined_features - text_features).norm(dim=-1)  # [B]
    label_norm = label_features.norm(dim=-1)  # [B]

    # Two sides: label too small or too big compared to delta
    low = F.relu(delta - label_norm - tau)
    high = F.relu(label_norm - delta - tau)
    return alpha * (low.pow(2) + high.pow(2)).mean()
