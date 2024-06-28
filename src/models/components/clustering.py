import numpy as np
import torch

from dask_cuda import LocalCUDACluster
from cuml import KMeans
from cuml.cluster import KMeans
from dask.distributed import Client
import dask.array as da
from cuml.datasets import make_blobs
from cuml.manifold import UMAP
from cuml.dask.manifold import UMAP as MNMG_UMAP
from cuml import DBSCAN, HDBSCAN
from cuml.cluster import DBSCAN, HDBSCAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Clustering:
    def __init__(self, embedding_manager, device="cuda"):
        self.embedding_manager = embedding_manager
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cluster = None
        self.client = None

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

    def merge_embeddings(self, umap_labels, centers, original_embeddings, update_type='hard', alpha=0.1):
        merged_embeddings = []
        unique_labels = torch.unique(umap_labels)
        
        for label in unique_labels:
            label_indices = torch.where(umap_labels == label)[0]
            # Collect original embeddings for the current cluster
            cluster_embeddings = torch.stack([original_embeddings[idx] for idx in label_indices])
            # Compute the centroid in the original 512-dimensional space
            centroid = cluster_embeddings.mean(dim=0)
            
            for idx in label_indices:
                idx = idx.item()  # Ensure idx is an integer
                sample_id = int(idx)  # Convert to integer
                if sample_id not in self.embedding_manager.index_mapping:
                    print(f"Index {sample_id} not found in index_mapping")
                    raise KeyError(f"Index {sample_id} not found in index_mapping")
                
                if update_type == 'hard':
                    # Hard update: Replace embedding with centroid
                    new_embedding = centroid
                elif update_type == 'soft':
                    # Soft update: Move embedding a bit towards the centroid
                    original_embedding = original_embeddings[sample_id]
                    new_embedding = original_embedding + alpha * (centroid - original_embedding)
                elif update_type == 'adaptive':
                    # Adaptive soft update: Alpha based on distance to centroid
                    original_embedding = original_embeddings[sample_id]
                    distance = torch.norm(centroid - original_embedding)
                    adaptive_alpha = alpha * (1 - torch.exp(-distance))  # Adaptive alpha
                    new_embedding = original_embedding + adaptive_alpha * (centroid - original_embedding)
                else:
                    raise ValueError("Invalid update_type. Choose 'hard', 'soft', or 'adaptive'.")

                self.embedding_manager.update_embedding(sample_id, new_embedding)


def main():

    # Example usage

    from src.data.imp_datamodule import EmbeddingManager, FolderManager
    import json

    # Load annotations
    annotation_path_train = (
        "/data/SSD/flickr8k/annotations/train.json"  # 1 caption per image for flickr8k
    )
    image_path = "/data/SSD/flickr8k/Images"
    keyword = "clip-small"

    # Initialize FolderManager
    folder_manager = FolderManager(base_log_dir="/project/Deep-Clustering/res")

    # Initialize experiment
    experiment_dir = folder_manager.initialize_experiment(keyword)

    # Initialize embedding manager
    annotations = json.load(open(annotation_path_train))
    embedding_manager = EmbeddingManager(
        annotations, embedding_dim=512, chunk_size=10000, hdf5_dir=experiment_dir
    )
    clustering = Clustering(embedding_manager)

    label_embedding = torch.randn(1000, 512).to(clustering.device)  # Dummy data
    umap_features = clustering.get_umap(label_embedding)
    umap_labels, centers = clustering.get_kmeans(umap_features, n_clusters=10)

    # Merge embeddings
    merged_embeddings = clustering.merge_embeddings(umap_labels, centers)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(umap_features[:, 0], umap_features[:, 1], c=umap_labels, s=0.1)
    # output the figure
    plt.savefig("plot/umap.png")


if __name__ == "__main__":
    main()
