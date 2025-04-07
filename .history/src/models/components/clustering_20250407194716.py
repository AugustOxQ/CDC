# import numpy as np
import random
from collections import defaultdict
from cProfile import label

import dask.array as da
import numpy as np
import torch
from cuml.cluster import HDBSCAN, KMeans
from cuml.dask.manifold import UMAP as MNMG_UMAP
from cuml.datasets import make_blobs
from cuml.manifold import UMAP
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from scipy import cluster
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UMAP_vis:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cluster = None
        self.client = None
        self.local_model = None  # Save the model

    def initialize_cluster(self):
        # Initialize Dask CUDA cluster and client
        self.cluster = LocalCUDACluster(threads_per_worker=1)
        self.client = Client(self.cluster)

    def close_cluster(self):
        # Close Dask CUDA cluster and client
        self.client.close()  # type: ignore
        self.cluster.close()  # type: ignore

    def learn_umap(self, embedding, n_components: int = 2):
        # Perform UMAP dimensionality reduction on embeddings
        self.initialize_cluster()

        embedding = embedding.to(self.device)
        label_embedding_np = embedding.cpu().numpy()

        local_model = UMAP(random_state=42, n_components=n_components)
        umap_features = local_model.fit_transform(label_embedding_np)

        # Save the model
        if self.local_model is None:
            self.local_model = local_model

        return umap_features

    def predict_umap(self, new_embedding: np.ndarray):
        if self.local_model is None:
            raise ValueError("UMAP model has not been trained yet.")
        umap_features_new = self.local_model.transform(new_embedding)
        return umap_features_new


class Clustering:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cluster = None
        self.client = None

    def initialize_cluster(self):
        # Initialize Dask CUDA cluster and client
        self.cluster = LocalCUDACluster(threads_per_worker=1)
        self.client = Client(self.cluster)

    def close_cluster(self):
        # Close Dask CUDA cluster and client
        self.client.close()  # type: ignore
        self.cluster.close()  # type: ignore

    def get_umap(self, label_embedding, n_components: int = 2):
        # Perform UMAP dimensionality reduction on embeddings
        self.initialize_cluster()

        label_embedding = label_embedding.to(self.device)
        label_embedding_np = label_embedding.cpu().numpy()

        local_model = UMAP(random_state=42, n_components=n_components)
        umap_features = local_model.fit_transform(label_embedding_np)

        self.close_cluster()

        umap_features = torch.tensor(umap_features, device=self.device)
        return umap_features

    def get_and_predict_umap(
        self, label_embedding, label_embeddings_new=None, n_components: int = 2
    ):
        self.initialize_cluster()

        label_embedding = label_embedding.to(self.device)
        label_embedding_np = label_embedding.cpu().numpy()

        local_model = UMAP(random_state=42, n_components=n_components)
        umap_features = local_model.fit_transform(label_embedding_np)

        # Predict UMAP features for new embeddings
        if label_embeddings_new is not None:
            label_embeddings_new = label_embeddings_new.to(self.device)
            label_embeddings_new_np = label_embeddings_new.cpu().numpy()
            umap_features_new = local_model.transform(label_embeddings_new_np)  # type: ignore

        self.close_cluster()

        umap_features = torch.tensor(umap_features, device=self.device)
        umap_features_new = torch.tensor(umap_features_new, device=self.device)

        return umap_features, umap_features_new

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

    def get_hdbscan(self, umap_features, n_clusters):
        self.initialize_cluster()

        umap_features_np = umap_features.cpu().numpy()
        hdbscan_model = HDBSCAN(
            min_cluster_size=100,
            min_samples=50,
            cluster_selection_method="leaf",  # https://docs.rapids.ai/api/cuml/stable/api/#hdbscan
        )
        hdbscan_model.fit(umap_features_np)
        umap_labels = hdbscan_model.labels_

        self.close_cluster()

        umap_labels = torch.tensor(umap_labels, device=self.device)
        centers = None

        return umap_labels, None

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

        print("Updating embeddings")
        if update_type == "hard":
            # Hard update: Directly replace embeddings with their corresponding cluster centers
            for i in tqdm(range(num_clusters)):
                cluster_indices = (
                    (umap_labels == i).nonzero(as_tuple=True)[0].to(original_embeddings.device)
                )
                updated_embeddings[cluster_indices] = cluster_centers[i]

        return updated_embeddings

    def hdbscan_update(
        self,
        umap_labels,
        original_embeddings,
        update_type="hard",
        alpha=0.5,
        update_noise="ignore",  # 'ignore' or 'assign'
    ):
        device = original_embeddings.device

        # clip alpha between 0.01 and 0.99
        alpha = max(min(alpha, 0.99), 0.01)

        # Exclude noise points labeled as -1
        unique_labels = umap_labels.unique()
        non_noise_labels = unique_labels[unique_labels != -1]
        num_clusters = len(non_noise_labels)

        cluster_centers = torch.zeros(
            (num_clusters, original_embeddings.shape[1]),
            device=device,
        )
        updated_embeddings = original_embeddings.clone()

        # Map labels to indices for cluster centers
        label_to_idx = {nn_label.item(): idx for idx, nn_label in enumerate(non_noise_labels)}

        # Calculate the mean of the embeddings for each cluster
        print("Calculating cluster centers")
        for non_noise_label in tqdm(non_noise_labels):
            cluster_indices = (umap_labels == non_noise_label).nonzero(as_tuple=True)[0].to(device)
            cluster_embeddings = original_embeddings[cluster_indices]
            cluster_center = cluster_embeddings.mean(dim=0)
            cluster_centers[label_to_idx[non_noise_label.item()]] = cluster_center

        if update_noise == "assign":
            print("Assigning noise points to nearest clusters")
            noise_indices = (umap_labels == -1).nonzero(as_tuple=True)[0].to(device)
            if len(noise_indices) > 0:
                noise_embeddings = original_embeddings[noise_indices].to(device)
                # Compute distances between noise embeddings and cluster centers
                distances = torch.cdist(noise_embeddings, cluster_centers)
                # Assign noise points to the nearest cluster
                nearest_clusters = distances.argmin(dim=1)
                # Convert nearest cluster labels to the correct device
                assigned_labels = torch.tensor(
                    [non_noise_labels[cluster_idx].item() for cluster_idx in nearest_clusters],
                    device=device,
                    dtype=umap_labels.dtype,
                )

                # Assign the labels to umap_labels
                umap_labels[noise_indices] = assigned_labels.to(
                    umap_labels.device
                )  # Ensure umap_labels is also on the same device
        elif update_noise == "ignore":
            # Do not update noise points
            pass
        else:
            raise ValueError("update_noise must be 'ignore' or 'assign'.")

        # Update embeddings
        print("Updating embeddings")
        if update_type == "hard":
            # Hard update: Replace embeddings with their corresponding cluster centers
            for non_noise_label in tqdm(unique_labels):
                cluster_indices = (
                    (umap_labels == non_noise_label).nonzero(as_tuple=True)[0].to(device)
                )
                if non_noise_label == -1:
                    if update_noise == "assign":
                        # Noise points have been assigned to clusters
                        continue
                    else:
                        # Do not update noise points
                        continue
                cluster_center = cluster_centers[label_to_idx[non_noise_label.item()]]
                updated_embeddings[cluster_indices] = cluster_center
        elif update_type == "soft":
            for non_noise_label in tqdm(non_noise_labels):
                cluster_indices = (
                    (umap_labels == non_noise_label)
                    .nonzero(as_tuple=True)[0]
                    .to(original_embeddings.device)
                )
                if non_noise_label == -1:
                    if update_noise == "assign":
                        # Noise points have been assigned to clusters
                        continue
                    else:
                        # Do not update noise points
                        continue
                updated_embeddings[cluster_indices] = (1 - alpha) * original_embeddings[
                    cluster_indices
                ] + alpha * cluster_centers[label_to_idx[non_noise_label.item()]]
        else:
            raise ValueError("update_type must be 'hard' or 'soft'.")

        if len(cluster_centers) == 0:
            cluster_centers = torch.zeros((1, original_embeddings.shape[-1]), device=device)
        return updated_embeddings, cluster_centers


def test_kmeans():
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


def test_hdbscan():
    # Example usage of Clustering class
    random.seed(42)

    clustering = Clustering(device="cuda")

    # label_embedding = torch.randn(1024, 512).to(clustering.device)  # Dummy data

    num_clusters = 5
    num_points_per_cluster = 1000
    dim = 512

    clusters = []

    for k in range(num_clusters):
        # Generate a random mean vector for each cluster
        # Offset means to ensure clusters are well-separated
        mu_k = torch.randn(dim) * 10 + k * 100.0

        # Generate samples for the cluster0
        samples = torch.randn(num_points_per_cluster, dim) + mu_k
        clusters.append(samples)

    # Combine all clusters into one dataset
    label_embedding = torch.cat(clusters, dim=0).to(clustering.device)

    umap_features = clustering.get_umap(label_embedding)

    umap_labels, centers = clustering.get_hdbscan(umap_features, n_clusters=1024 - 30)
    print(umap_labels.shape)

    print(type(umap_labels))

    updated_embeddings, centers = clustering.hdbscan_update(
        umap_labels,
        label_embedding,
        update_type="hard",
        alpha=0.1,
        update_noise="ignore",
    )

    # Check if embeddings have been updated
    differences = torch.any(label_embedding != updated_embeddings, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)

    umap_features_new = clustering.get_and_predict_umap(updated_embeddings)
    umap_labels_new, centers_new = clustering.get_hdbscan(umap_features_new, n_clusters=1024 - 60)

    updated_embeddings_new, centers_new = clustering.hdbscan_update(
        umap_labels_new,
        updated_embeddings,
        update_type="hard",
        alpha=0.1,
        update_noise="assign",
    )

    print(f"Centers: {centers_new.shape}")

    # Check if embeddings have been updated
    differences = torch.any(updated_embeddings != updated_embeddings_new, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)


def main():
    test_hdbscan()


if __name__ == "__main__":
    main()
