import torch


def boundary_penalty(embeddings, radius=1.0, alpha=0.1):
    norms = torch.norm(embeddings, p=2, dim=1)
    penalty = torch.where(norms > radius, (norms - radius) ** 2, torch.zeros_like(norms))
    return alpha * torch.mean(penalty)


def l2_regularizer(embeddings, alpha=0.1):
    l2_norm = torch.norm(embeddings, p=2, dim=1)  # Compute L2 norm for each embedding
    return alpha * torch.mean(l2_norm**2)  # Return the mean L2 norm with scaling
