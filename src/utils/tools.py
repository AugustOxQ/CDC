import os

import numpy as np
import matplotlib.pyplot as plt

def calculate_n_clusters(initial_n_clusters, first_stage_n, second_stage_n, epoch, k_means_start_epoch = 10, k_means_slow_epoch = 40, decay_rate = 0.8):
    
    if epoch < k_means_start_epoch:
        # Not doing any clustering
        return 0
    
    if k_means_start_epoch <= epoch < k_means_slow_epoch:
        
        adjusted_total_epochs = k_means_slow_epoch - k_means_start_epoch
        adjusted_epoch = epoch - k_means_start_epoch
        
        # Sigmoid function to control the rate of decrease
        x = np.linspace(-6, 6, adjusted_total_epochs)  # Spread over a range to adjust the curve
        sigmoid = 1 / (1 + np.exp(-x))

        # Normalize to get values between 0 and 1
        normalized_sigmoid = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())

        # Invert and scale to get the number of clusters
        n_clusters = initial_n_clusters - normalized_sigmoid * (
            initial_n_clusters - first_stage_n
        )
        
        return int(n_clusters[adjusted_epoch])
    
    else:
        # From P to total_epochs, decay by 0.9 per epoch
        n_clusters = int(first_stage_n * (decay_rate ** (epoch - k_means_slow_epoch + 1)))
        n_clusters = max(n_clusters, second_stage_n)

        return n_clusters


def plot_umap(umap_features_np, umap_labels, plot_dir, epoch, samples_to_track=[]):
    # Plot UMAP before clustering update
    fig = plt.figure(figsize=(16, 16))
    plt.scatter(umap_features_np[:, 0], umap_features_np[:, 1], c=umap_labels, s=0.1)

    # Highlight and annotate the tracked samples
    for sample_idx in samples_to_track:
        x, y = umap_features_np[sample_idx, :]
        plt.scatter(x, y, c="red", s=150, edgecolors="k")  # Highlight the sample
        plt.text(
            x, y, f"Sample {sample_idx}", fontsize=24, color="black"
        )  # Annotate the sample

    # Add the number of umap_labels to the plot as title
    plt.title(f"UMAP with {len(umap_labels)} clusters")
    
    plt.colorbar()
    # output the figure
    plt.savefig(os.path.join(plot_dir, f"umap_{epoch}.png"))
    plt.close(fig)