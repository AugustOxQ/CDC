# import numpy as np
import random
from collections import defaultdict

import dask.array as da
import numpy as np
import torch

# from cuml import DBSCAN, HDBSCAN, KMeans
from cuml.cluster import DBSCAN, HDBSCAN, KMeans
from cuml.dask.manifold import UMAP as MNMG_UMAP
from cuml.datasets import make_blobs
from cuml.manifold import UMAP
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Clustering:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cluster = None
        self.client = None
        # self.cluster_history = defaultdict(set)

    def initialize_cluster(self):
        # Initialize Dask CUDA cluster and client
        self.cluster = LocalCUDACluster(threads_per_worker=1)
        self.client = Client(self.cluster)

    def close_cluster(self):
        # Close Dask CUDA cluster and client
        self.client.close()
        self.cluster.close()

    def get_umap(self, label_embedding):
        # Perform UMAP dimensionality reduction on embeddings
        self.initialize_cluster()

        label_embedding = label_embedding.to(self.device)
        label_embedding_np = label_embedding.cpu().numpy()

        local_model = UMAP(random_state=42)
        umap_features = local_model.fit_transform(label_embedding_np)

        self.close_cluster()

        umap_features = torch.tensor(umap_features, device=self.device)
        return umap_features

    def get_kmeans(self, umap_features, n_clusters):
        # Perform KMeans clustering on UMAP features
        self.initialize_cluster()

        umap_features_np = umap_features.cpu().numpy()
        k_means = KMeans(n_clusters=n_clusters, random_state=42).fit(umap_features_np)
        umap_labels = k_means.labels_
        centers = k_means.cluster_centers_

        self.close_cluster()

        umap_labels = torch.tensor(umap_labels, device=umap_features.device)
        centers = torch.tensor(centers, device=umap_features.device)

        return umap_labels, centers

    def get_hdbscan(
        self,
        umap_features,
    ):
        self.initialize_cluster()

        umap_features_np = umap_features.cpu().numpy()
        hdbscan = HDBSCAN(min_samples=10, min_cluster_size=500)
        hdbscan.fit(umap_features_np)
        umap_labels = hdbscan.labels_

        self.close_cluster()

        umap_features = torch.tensor(umap_features, device=self.device)
        centers = torch.zeros(umap_features.size(1), device=self.device)

        return umap_labels, centers

    def kmeans_update(
        self,
        umap_labels,
        original_embeddings,
        update_type="hard",
        alpha=0.1,
        repulsion_factor=0.05,
        random_repulsion=False,
        threshold_k=1000,
    ):
        num_clusters = umap_labels.max().item() + 1
        cluster_centers = torch.zeros(
            (num_clusters, original_embeddings.shape[1]),
            device=original_embeddings.device,
        )
        updated_embeddings = torch.zeros_like(original_embeddings)

        # Calculate the mean of the embeddings for each cluster
        print("Calculating cluster centers")
        for i in tqdm(range(num_clusters)):
            cluster_indices = (
                (umap_labels == i).nonzero(as_tuple=True)[0].to(original_embeddings.device)
            )
            cluster_embeddings = original_embeddings[cluster_indices]
            cluster_centers[i] = cluster_embeddings.mean(dim=0)

        # # Repulsion force to make clusters far away from each other
        # if threshold_k < num_clusters < 5000:
        #     if repulsion_factor > 0:
        #         print("Applying repulsion force")
        #         if random_repulsion:
        #             # Randomly select 1/4 of the centroids for repulsion
        #             sampled_indices = random.sample(range(num_clusters), num_clusters // 4)
        #             for i in tqdm(sampled_indices):  # Iterate over only the sampled centroids
        #                 for j in range(i + 1, num_clusters):
        #                     diff = cluster_centers[i] - cluster_centers[j]
        #                     distance = diff.norm(p=2)
        #                     if distance > 0:  # Avoid division by zero
        #                         repulsion = repulsion_factor * diff / distance
        #                         cluster_centers[i] += repulsion
        #                         cluster_centers[j] -= repulsion
        #         else:
        #             # Apply repulsion force to all centroids
        #             for i in tqdm(range(num_clusters)):
        #                 for j in range(i + 1, num_clusters):
        #                     diff = cluster_centers[i] - cluster_centers[j]
        #                     distance = diff.norm(p=2)
        #                     if distance > 0:  # Avoid division by zero
        #                         repulsion = repulsion_factor * diff / distance
        #                         cluster_centers[i] += repulsion
        #                         cluster_centers[j] -= repulsion
        # else: # If K is small, apply selective elimination of clusters
        print(
            f"Selective elimination of clusters as K = {num_clusters} is small"
        )  # Currently only select this method
        # Step 1: Compute the variance and size of each cluster
        cluster_variances = torch.zeros(num_clusters, device=original_embeddings.device)
        cluster_sizes = torch.zeros(num_clusters, device=original_embeddings.device)

        for i in range(num_clusters):
            cluster_indices = (
                (umap_labels == i).nonzero(as_tuple=True)[0].to(original_embeddings.device)
            )
            cluster_embeddings = original_embeddings[cluster_indices]
            cluster_sizes[i] = cluster_embeddings.size(0)  # Cluster size
            if cluster_embeddings.size(0) > 1:
                cluster_variances[i] = cluster_embeddings.var(
                    dim=0
                ).mean()  # Average variance per cluster

        # Step 2: Rank clusters by some criterion (size and variance)
        # Select a combination of size and variance to eliminate less meaningful clusters
        combined_score = cluster_variances + (
            1 / cluster_sizes
        )  # Small clusters and high variance clusters will have higher scores

        # Keep clusters with the lowest combined score and eliminate the others
        num_clusters_to_keep = max(1, num_clusters // 2)  # For example, keep half the clusters
        _, keep_indices = combined_score.topk(
            num_clusters_to_keep, largest=False
        )  # Keep clusters with the lowest score

        # Step 3: Replace eliminated clusters with the centroid of the kept cluster(s)
        for i in tqdm(range(num_clusters)):
            cluster_indices = (
                (umap_labels == i).nonzero(as_tuple=True)[0].to(original_embeddings.device)
            )
            if i in keep_indices:
                # This is a kept cluster, so update embeddings normally
                updated_embeddings[cluster_indices] = cluster_centers[i]
            else:
                # This is an eliminated cluster, replace with centroid of a kept cluster
                chosen_cluster_idx = keep_indices[
                    0
                ]  # For simplicity, replace with the first kept cluster centroid
                updated_embeddings[cluster_indices] = cluster_centers[chosen_cluster_idx]

        print("Updating embeddings")
        if update_type == "hard":
            # Hard update: Directly replace embeddings with their corresponding cluster centers
            for i in tqdm(range(num_clusters)):
                cluster_indices = (
                    (umap_labels == i).nonzero(as_tuple=True)[0].to(original_embeddings.device)
                )
                updated_embeddings[cluster_indices] = cluster_centers[i]
        # elif update_type == "soft":
        #     # Soft update: Move embeddings towards their corresponding cluster centers by a factor of alpha
        #     for i in tqdm(range(num_clusters)):
        #         cluster_indices = (
        #             (umap_labels == i)
        #             .nonzero(as_tuple=True)[0]
        #             .to(original_embeddings.device)
        #         )
        #         updated_embeddings[cluster_indices] += alpha * (
        #             cluster_centers[i] - original_embeddings[cluster_indices]
        #         )

        return updated_embeddings


def main():
    # Example usage of Clustering class
    random.seed(42)

    clustering = Clustering(device="cuda")

    label_embedding = torch.randn(1024, 512).to(clustering.device)  # Dummy data

    umap_features = clustering.get_umap(label_embedding)

    umap_labels, centers = clustering.get_kmeans(umap_features, n_clusters=1024 - 30)

    updated_embeddings = clustering.kmeans_update(
        umap_labels, label_embedding, update_type="hard", alpha=0.1
    )

    # Check if embeddings have been updated
    differences = torch.any(label_embedding != updated_embeddings, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)

    umap_features_new = clustering.get_umap(updated_embeddings)
    umap_labels_new, centers_new = clustering.get_kmeans(umap_features_new, n_clusters=1024 - 60)

    updated_embeddings_new = clustering.kmeans_update(
        umap_labels_new, updated_embeddings, update_type="hard", alpha=0.1
    )

    # Check if embeddings have been updated
    differences = torch.any(updated_embeddings != updated_embeddings_new, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)


if __name__ == "__main__":
    main()
