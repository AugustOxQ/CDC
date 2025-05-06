import torch
import torch.nn.functional as F


def boundary_penalty(embeddings, radius=1.0, alpha=0.1):
    norms = torch.norm(embeddings, p=2, dim=1)
    penalty = torch.where(norms > radius, (norms - radius) ** 2, torch.zeros_like(norms))
    return alpha * torch.mean(penalty)


def l2_regularizer(embeddings, alpha=0.1):
    l2_norm = torch.norm(embeddings, p=2, dim=1)  # Compute L2 norm for each embedding
    return alpha * torch.mean(l2_norm**2)  # Return the mean L2 norm with scaling


def text_preserve_regularizer(text_features, combined_features, tau=0.3, alpha=0.1):
    delta = (combined_features - text_features).norm(dim=-1)  # [B]
    excess_change = F.relu(delta - tau)
    return alpha * excess_change.pow(2).mean()


def label_change_regularizer(text_features, combined_features, label_features, tau=0.3, alpha=0.1):
    delta = (combined_features - text_features).norm(dim=-1)  # [B]
    label_norm = label_features.norm(dim=-1)  # [B]

    # Two sides: label too small or too big compared to delta
    low = F.relu(delta - label_norm - tau)
    high = F.relu(label_norm - delta - tau)
    return alpha * (low.pow(2) + high.pow(2)).mean()


def pull_away_diversity_loss(label_proj, alpha=0.1):
    normalized = F.normalize(label_proj, dim=-1)
    sim_matrix = torch.matmul(normalized, normalized.T)
    batch_size = label_proj.size(0)
    mask = torch.eye(batch_size, device=label_proj.device).bool()
    off_diag = sim_matrix[~mask].view(batch_size, -1)
    return alpha * (off_diag**2).mean()


def angular_consistency_loss(label_proj, text_features, combined_features, alpha=0.1):
    delta = F.normalize(combined_features - text_features, dim=-1)
    label_proj = F.normalize(label_proj, dim=-1)
    return alpha * (1 - (delta * label_proj).sum(dim=-1)).mean()


def entropy_loss(embeddings: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
    p = F.softmax(sim, dim=1)
    log_p = torch.log(p + 1e-6)
    entropy = -torch.sum(p * log_p, dim=1)
    return -alpha * entropy.mean()  # maximize entropy
