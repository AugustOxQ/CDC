from re import U

import geoopt
import matplotlib.pyplot as plt
import numpy as np
import torch
from geoopt.manifolds import stereographic as poincare
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HyperbolicProjectionAndClustering:
    def __init__(
        self,
        lr=1e-3,
        epochs=1000,
        k_clusters=5,
        max_iters=100,
        batch_size=1024,
        device=device,
    ):
        """Initialize the projection and clustering class with GPU support and batching.

        Args:
        lr: Learning rate for optimization.
        epochs: Number of epochs for projection.
        k_clusters: Number of clusters for k-means clustering.
        max_iters: Maximum iterations for the k-means algorithm.
        batch_size: Size of the mini-batches to process the data.
        device: The device to run computations on ('cpu' or 'cuda').
        """
        self.lr = lr
        self.epochs = epochs
        self.k_clusters = k_clusters
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.manifold = poincare.PoincareBall()
        self.hyperbolic_embeddings = None
        self.device = device

    def project_to_hyperbolic(self, euclidean_data):
        """Projects the Euclidean (UMAP) data into hyperbolic space using the Poincare Ball model.

        Args:
        euclidean_data: A tensor of shape [N, 2], where N is the number of samples (UMAP embeddings).

        Returns:
        hyperbolic_embeddings: A tensor of shape [N, 2], projected into hyperbolic space.
        """
        # Convert to torch tensor if not already
        if not torch.is_tensor(euclidean_data):
            euclidean_data = torch.tensor(euclidean_data).to(self.device)

        # Initialize hyperbolic embeddings close to the Euclidean data
        hyperbolic_embeddings = (
            euclidean_data.clone().detach() + torch.randn_like(euclidean_data) * 1e-1
        )
        hyperbolic_embeddings.requires_grad = True

        # Move hyperbolic embeddings to the device
        hyperbolic_embeddings = hyperbolic_embeddings.to(self.device)

        # Define optimizer
        optimizer = torch.optim.Adam([hyperbolic_embeddings], lr=self.lr)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.epochs, T_mult=self.epochs, eta_min=1e-9
        )

        # Prepare data loader for batching
        dataset = TensorDataset(euclidean_data, hyperbolic_embeddings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss function (for preserving distance)
        # def hyperbolic_loss(euclidean_batch, hyperbolic_batch):
        #     dist_euclidean = torch.cdist(euclidean_batch, euclidean_batch)
        #     dist_hyperbolic = self.manifold.dist(hyperbolic_batch, hyperbolic_batch)
        #     dist_euclidean_normalized = dist_euclidean / dist_euclidean.max()
        #     dist_hyperbolic_normalized = dist_hyperbolic / dist_hyperbolic.max()
        #     loss = torch.norm(dist_euclidean_normalized - dist_hyperbolic_normalized)
        #     return loss

        def hyperbolic_loss(euclidean_data, hyperbolic_data):
            # Euclidean distance
            dist_euclidean = torch.cdist(euclidean_data, euclidean_data)

            # Hyperbolic distance (using the manifold distance function)
            dist_hyperbolic = self.manifold.dist(hyperbolic_data, hyperbolic_data)

            # Normalizing the distances
            dist_euclidean_normalized = dist_euclidean / dist_euclidean.max()
            dist_hyperbolic_normalized = dist_hyperbolic / dist_hyperbolic.max()

            # Loss that tries to preserve relative distances
            loss = torch.norm(dist_euclidean_normalized - dist_hyperbolic_normalized)
            return loss

        # Optimization loop
        for epoch in tqdm(range(self.epochs)):
            total_loss = 0.0
            for batch_idx, (euclidean_batch, hyperbolic_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = hyperbolic_loss(euclidean_batch, hyperbolic_batch)
                loss.backward()  # Backpropagation
                torch.nn.utils.clip_grad_norm_([hyperbolic_embeddings], max_norm=0.5)
                optimizer.step()
                scheduler.step(epoch + batch_idx / len(dataloader))  # type: ignore
                total_loss += loss.item()

            # Optional: print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

        # Store the hyperbolic embeddings in the class
        self.hyperbolic_embeddings = hyperbolic_embeddings
        return hyperbolic_embeddings

    def hyperbolic_kmeans(self):
        """Perform k-means clustering in hyperbolic space.

        Returns:
        cluster_assignments: A tensor of shape [N], cluster labels for each data point.
        centroids: A tensor of shape [k_clusters, 2], the cluster centroids in hyperbolic space.
        """
        if self.hyperbolic_embeddings is None:
            raise ValueError("You must project the data to hyperbolic space before clustering.")

        data = self.hyperbolic_embeddings
        k = self.k_clusters

        # Initialize centroids randomly
        centroids = data[torch.randint(0, data.shape[0], (k,))]

        # Prepare data loader for batching during clustering
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False)  # type: ignore

        for i in range(self.max_iters):
            # Batch assignment and centroid update
            distances = []
            cluster_assignments = torch.empty(data.shape[0], dtype=torch.long, device=self.device)

            for batch_idx, batch_data in enumerate(dataloader):
                batch_distances = torch.stack(
                    [self.manifold.dist(batch_data, centroid) for centroid in centroids],
                    dim=1,
                )
                batch_assignments = torch.argmin(batch_distances, dim=1)
                cluster_assignments[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ] = batch_assignments

            # Recompute centroids as the mean of assigned points in each cluster
            new_centroids = []
            for j in range(k):
                points_in_cluster = data[cluster_assignments == j]
                if len(points_in_cluster) > 0:
                    new_centroids.append(self.manifold.projx(points_in_cluster.mean(dim=0)))
                else:
                    # If a centroid has no points, reinitialize it to a random point from the dataset
                    random_centroid = data[torch.randint(0, data.shape[0], (1,))]
                    new_centroids.append(
                        random_centroid.squeeze(0)
                    )  # Ensure the shape is correct ([2])

            new_centroids = torch.stack(new_centroids)

            # Check convergence (if centroids do not move)
            if torch.allclose(centroids, new_centroids, atol=1e-3):
                break

            centroids = new_centroids

        return cluster_assignments, centroids

    def fit(self, euclidean_data):
        """Projects the Euclidean (UMAP) data to hyperbolic space and performs hyperbolic k-means
        clustering.

        Args:
        euclidean_data: A tensor or array of shape [N, 2].

        Returns:
        cluster_assignments: A tensor of shape [N], the cluster labels.
        centroids: A tensor of shape [k_clusters, 2], the centroids in hyperbolic space.
        """
        self.project_to_hyperbolic(euclidean_data)
        return self.hyperbolic_kmeans()


def visualize_clusters(umap_embeddings, cluster_assignments, centroids):
    """Visualize the clusters and centroids.

    Args:
    umap_embeddings: A tensor or numpy array of shape [N, 2] (the original UMAP embeddings).
    cluster_assignments: A tensor of shape [N] with the cluster labels for each point.
    centroids: A tensor or numpy array of shape [k_clusters, 2], the centroids of the clusters.
    """
    # Convert tensors to numpy for compatibility with matplotlib
    if torch.is_tensor(umap_embeddings):
        umap_embeddings = umap_embeddings.detach().cpu().numpy()
    if torch.is_tensor(cluster_assignments):
        cluster_assignments = cluster_assignments.detach().cpu().numpy()
    if torch.is_tensor(centroids):
        centroids = centroids.detach().cpu().numpy()

    # Number of clusters
    num_clusters = len(centroids)

    # Plot each cluster with a different color
    plt.figure(figsize=(10, 8))

    for cluster_id in range(num_clusters):
        points_in_cluster = umap_embeddings[cluster_assignments == cluster_id]
        plt.scatter(
            points_in_cluster[:, 0],
            points_in_cluster[:, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
        )

    # Plot the centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="red",
        marker="x",
        s=100,
        label="Centroids",
    )

    # Add title, legend, and axis labels
    plt.title("Clusters and Centroids Visualization")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("tmp/clusters_and_centroids.png")
    plt.show()


def test_hyperbolic_clustering(umap_embeddings):
    # Usage Example:
    # umap_embeddings = your_umap_embeddings  # [N, 2]

    # Create an instance of the class
    hyperbolic_clustering = HyperbolicProjectionAndClustering(
        lr=1e-2,
        epochs=1000,
        k_clusters=5,
        max_iters=100,
        batch_size=1024,
        device=device,
    )

    # Fit the model to the UMAP embeddings
    cluster_assignments, centroids = hyperbolic_clustering.fit(umap_embeddings)
    visualize_clusters(umap_embeddings, cluster_assignments, centroids)


def main():
    # generate umap data of 5 clusters using 5 different normaldistributions
    umap_1 = np.random.multivariate_normal(mean=[50, 50], cov=[[1, 0], [0, 1]], size=3000)
    umap_2 = np.random.multivariate_normal(mean=[10, 10], cov=[[1, 0], [0, 1]], size=3000)
    umap_3 = np.random.multivariate_normal(mean=[-10, -10], cov=[[1, 0], [0, 1]], size=3000)
    umap_4 = np.random.multivariate_normal(mean=[0, 10], cov=[[1, 0], [0, 1]], size=3000)
    umap_5 = np.random.multivariate_normal(mean=[0, -10], cov=[[1, 0], [0, 1]], size=3000)
    umap_embeddings = np.concatenate((umap_1, umap_2, umap_3, umap_4, umap_5))
    # Shuffle the data
    umap_embeddings = umap_embeddings[torch.randperm(umap_embeddings.shape[0])]

    print(umap_embeddings.shape)

    test_hyperbolic_clustering(umap_embeddings)


if __name__ == "__main__":
    main()
