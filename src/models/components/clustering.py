# import numpy as np
import torch

from dask_cuda import LocalCUDACluster
from cuml import KMeans
from cuml.cluster import KMeans
from dask.distributed import Client
from collections import defaultdict
# import dask.array as da
# from cuml.datasets import make_blobs
from cuml.manifold import UMAP
# from cuml.dask.manifold import UMAP as MNMG_UMAP
# from cuml import DBSCAN, HDBSCAN
# from cuml.cluster import DBSCAN, HDBSCAN

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
    
    # def update_clustering_history(self, umap_labels):
    #     new_history = defaultdict(set)
        
    #     for label in torch.unique(umap_labels):
    #         label_indices = torch.where(umap_labels == label)[0].tolist()
    #         for idx in label_indices:
    #             self.cluster_history[idx].add(label)
    #             new_history[label].update(self.cluster_history[idx])

    #     self.cluster_history = new_history

    # def merge_clusters(self, original_embeddings):
    #     merged_embeddings = []
    #     unique_clusters = sorted(self.cluster_history.keys())

    #     for cluster_id in unique_clusters:
    #         cluster_indices = list(self.cluster_history[cluster_id])
    #         cluster_embeddings = original_embeddings[cluster_indices]
    #         merged_embedding = cluster_embeddings.mean(dim=0)
    #         merged_embeddings.append(merged_embedding)

    #     return torch.stack(merged_embeddings), unique_clusters

    def kmeans_update(self, umap_labels, centers, original_embeddings, update_type='hard', alpha=0.1):
        updated_embeddings = torch.zeros_like(original_embeddings)
        unique_labels = torch.unique(umap_labels)
        
        for label in unique_labels:
            label_indices = torch.where(umap_labels == label)[0]
            
            if len(label_indices) == 1:
                idx = label_indices[0].item()
                updated_embeddings[idx] = original_embeddings[idx]
                continue
            
            # Collect original embeddings for the current cluster
            cluster_embeddings = original_embeddings[label_indices.cpu()]
            
            # Compute the centroid in the original 512-dimensional space
            centroid = cluster_embeddings.mean(dim=0)
            
            for idx in label_indices:
                idx = idx.item()  # Ensure idx is an integer
                
                if update_type == 'hard':
                    # Hard update: Replace embedding with centroid
                    new_embedding = centroid
                elif update_type == 'soft':
                    # Soft update: Move embedding a bit towards the centroid
                    original_embedding = original_embeddings[idx]
                    new_embedding = original_embedding + alpha * (centroid - original_embedding)
                elif update_type == 'adaptive':
                    # Adaptive soft update: Alpha based on distance to centroid
                    original_embedding = original_embeddings[idx]
                    distance = torch.norm(centroid - original_embedding)
                    adaptive_alpha = alpha * (1 - torch.exp(-distance))  # Adaptive alpha
                    new_embedding = original_embedding + adaptive_alpha * (centroid - original_embedding)
                else:
                    raise ValueError("Invalid update_type. Choose 'hard', 'soft', or 'adaptive'.")
                
                updated_embeddings[idx] = new_embedding
                
            # self.update_clustering_history(umap_labels)
                
        return updated_embeddings


def main():
    # Example usage of Clustering class
    import random 
    random.seed(42)
    
    clustering = Clustering(device = "cuda")

    label_embedding = torch.randn(1024, 512).to(clustering.device)  # Dummy data
    
    umap_features = clustering.get_umap(label_embedding)
    
    umap_labels, centers = clustering.get_kmeans(umap_features, n_clusters=1024-30)
    
    updated_embeddings = clustering.kmeans_update(umap_labels, centers, label_embedding, update_type='hard', alpha=0.1)
    
    # Check if embeddings have been updated
    differences = torch.any(label_embedding != updated_embeddings, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)

    umap_features_new = clustering.get_umap(updated_embeddings)
    umap_labels_new, centers_new = clustering.get_kmeans(umap_features_new, n_clusters=1024-60)
    
    updated_embeddings_new = clustering.kmeans_update(umap_labels_new, centers_new, updated_embeddings, update_type='hard', alpha=0.1)
    
    # Check if embeddings have been updated
    differences = torch.any(updated_embeddings != updated_embeddings_new, dim=1)
    num_different_rows = torch.sum(differences).item()
    print(num_different_rows)

if __name__ == "__main__":
    main()
