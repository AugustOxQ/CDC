# import geoopt
# import torch
# from geoopt import ManifoldTensor
# from geoopt.manifolds import PoincareBall
# from geoopt.manifolds.poincare.math import frechet_mean

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_hyperbolic(umap_features):
#     """Maps UMAP embeddings into hyperbolic space using the Poincaré disk model.

#     Args:
#         umap_features (torch.Tensor): UMAP embeddings of shape [N, 2].

#     Returns:
#         hyperbolic_embeddings (ManifoldTensor): Hyperbolic embeddings on the PoincareBall manifold.
#     """
#     # Ensure umap_features are in the Poincaré ball (norm less than 1)
#     norms = umap_features.norm(dim=-1, keepdim=True)
#     max_norm = norms.max()
#     if max_norm >= 1.0:
#         # Normalize to be within the unit disk
#         umap_features = (
#             umap_features / (max_norm + 1e-5) * 0.9999
#         )  # Slightly less than 1

#     # Define the PoincareBall manifold
#     manifold = PoincareBall(c=1.0)  # c is the negative curvature

#     # Create a ManifoldTensor on the PoincareBall manifold
#     hyperbolic_embeddings = ManifoldTensor(umap_features, manifold=manifold)

#     return hyperbolic_embeddings


# def hyperbolic_update(
#     hyperbolic_embeddings,
#     original_embeddings,
#     num_clusters,
#     update_type="hard",
#     alpha=0.1,
# ):
#     """Performs hyperbolic K-Means clustering and updates label embeddings.

#     Args:
#         hyperbolic_embeddings (ManifoldTensor): Hyperbolic embeddings on the PoincareBall manifold.
#         original_embeddings (torch.Tensor): Original label embeddings of shape [N, D].
#         num_clusters (int): Number of clusters for K-Means.
#         update_type (str): 'hard' or 'soft' update.
#         alpha (float): Mixing coefficient for 'soft' updates.

#     Returns:
#         updated_embeddings (torch.Tensor): Updated label embeddings.
#     """
#     # Perform hyperbolic K-Means clustering
#     N = hyperbolic_embeddings.shape[0]
#     device = hyperbolic_embeddings.device

#     # Randomly select initial centroids from the data points
#     indices = torch.randperm(N)[:num_clusters]
#     centroids = hyperbolic_embeddings[indices].clone()

#     manifold = hyperbolic_embeddings.manifold

#     for iteration in range(10):  # Number of iterations
#         # Assign points to nearest centroid
#         # Compute distances between points and centroids
#         # hyperbolic_embeddings: [N, 2]
#         # centroids: [num_clusters, 2]

#         # Expand dimensions for broadcasting
#         points_expanded = hyperbolic_embeddings.unsqueeze(1)  # [N, 1, 2]
#         centroids_expanded = centroids.unsqueeze(0)  # [1, num_clusters, 2]

#         # Compute hyperbolic distances
#         distances = manifold.dist(
#             points_expanded, centroids_expanded
#         )  # [N, num_clusters]

#         # Assign labels based on closest centroid
#         labels = distances.argmin(dim=1)  # [N]

#         # Update centroids
#         for k in range(num_clusters):
#             cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
#             if len(cluster_indices) == 0:
#                 # No points assigned to this centroid
#                 continue
#             cluster_points = hyperbolic_embeddings[cluster_indices]
#             # Compute Fréchet mean (hyperbolic centroid)
#             centroid = frechet_mean(
#                 cluster_points, manifold=manifold, lr=1e-3, max_iter=50
#             )
#             centroids[k] = centroid

#     # Update original embeddings based on clustering
#     updated_embeddings = original_embeddings.clone()

#     if update_type == "hard":
#         # Replace embeddings with the Euclidean centroid of their cluster
#         for k in range(num_clusters):
#             cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
#             if len(cluster_indices) == 0:
#                 continue
#             cluster_embeddings = original_embeddings[cluster_indices]
#             cluster_center = cluster_embeddings.mean(dim=0)
#             updated_embeddings[cluster_indices] = cluster_center
#     elif update_type == "soft":
#         # Move embeddings towards the cluster centroid
#         for k in range(num_clusters):
#             cluster_indices = (labels == k).nonzero(as_tuple=True)[0]
#             if len(cluster_indices) == 0:
#                 continue
#             cluster_embeddings = original_embeddings[cluster_indices]
#             cluster_center = cluster_embeddings.mean(dim=0)
#             updated_embeddings[cluster_indices] = (
#                 1 - alpha
#             ) * cluster_embeddings + alpha * cluster_center
#     else:
#         raise ValueError("update_type must be 'hard' or 'soft'.")

#     return updated_embeddings

# def
