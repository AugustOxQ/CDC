import os
import test

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def calculate_n_clusters(
    initial_n_clusters=1000,
    first_stage_n=2000,
    second_stage_n=50,
    epoch=30,
    k_means_start_epoch=10,
    k_means_slow_epoch=40,
    decay_rate=0.8,
):
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
        n_clusters = initial_n_clusters - normalized_sigmoid * (initial_n_clusters - first_stage_n)

        return int(n_clusters[adjusted_epoch])

    else:
        # From P to total_epochs, decay by 0.9 per epoch
        n_clusters = int(first_stage_n * (decay_rate ** (epoch - k_means_slow_epoch + 1)))
        n_clusters = max(n_clusters, second_stage_n)

        return n_clusters


def calculate_n_clusters_2(
    initial_n_clusters=3000,
    first_stage_n=200,
    second_stage_n=50,
    epoch=30,
    k_means_start_epoch=5,
    k_means_slow_epoch=20,
    k_means_end_epoch=25,
    decay_rate=0.8,
):
    n_clusters_list = []

    for epoch in range(epoch):
        if epoch < k_means_start_epoch:
            # First stage: no clustering
            n_clusters_list.append(0)

        elif k_means_start_epoch <= epoch < k_means_slow_epoch:
            # Second stage: Staircase-style drop every 2 epochs
            adjusted_epoch = epoch - k_means_start_epoch

            # Number of drops (every 2 epochs)
            num_steps = (k_means_slow_epoch - k_means_start_epoch) // 2
            step_size = (initial_n_clusters - first_stage_n) / num_steps

            if adjusted_epoch % 2 == 0:
                # Drop every two epochs
                n_clusters = initial_n_clusters - (adjusted_epoch // 2) * step_size
            else:
                # Stay the same on the odd epochs
                n_clusters = initial_n_clusters - (adjusted_epoch // 2) * step_size

            n_clusters_list.append(
                int(max(n_clusters, first_stage_n))
            )  # Ensure it's at least `first_stage_n`

        elif k_means_slow_epoch <= epoch < k_means_end_epoch:
            # Third stage: Decay by 0.9 per epoch
            n_clusters = int(first_stage_n * (decay_rate ** (epoch - k_means_slow_epoch + 1)))
            n_clusters = max(
                n_clusters, second_stage_n
            )  # Ensure it doesn't go below `second_stage_n`
            n_clusters_list.append(n_clusters)
        else:
            # Fourth stage: Final clustering
            n_clusters = second_stage_n
            n_clusters_list.append(n_clusters)

    return n_clusters_list


def calculate_n_clusters_3(
    initial_n_clusters=3000,
    first_stage_n=200,
    second_stage_n=50,
    epoch=30,
    k_means_start_epoch=5,
    k_means_slow_epoch=20,
    k_means_end_epoch=25,
    decay_rate=0.8,
):
    n_clusters_list = []

    for epoch in range(epoch):
        if epoch < k_means_start_epoch:
            # First stage: no clustering
            n_clusters_list.append(0)

        elif k_means_start_epoch <= epoch < k_means_end_epoch:
            # Second stage: Staircase-style drop every 2 epochs
            adjusted_epoch = epoch - k_means_start_epoch

            # Number of drops (every 2 epochs)
            num_steps = (k_means_end_epoch - k_means_start_epoch) // 5
            step_size = (initial_n_clusters - second_stage_n) / num_steps

            if adjusted_epoch % 5 == 0:
                # Drop every two epochs
                n_clusters = initial_n_clusters - (adjusted_epoch // 5) * step_size
            else:
                # Stay the same on the odd epochs
                n_clusters = initial_n_clusters - (adjusted_epoch // 5) * step_size

            n_clusters_list.append(
                int(max(n_clusters, second_stage_n))
            )  # Ensure it's at least `first_stage_n`
        else:
            # Fourth stage: Final clustering
            n_clusters = second_stage_n
            n_clusters_list.append(n_clusters)

    return n_clusters_list


def test_calculate_n_clusters_3():
    n_clusters_list = calculate_n_clusters_3(
        initial_n_clusters=200,
        first_stage_n=50,
        second_stage_n=5,
        epoch=30,
        k_means_start_epoch=5,
        k_means_slow_epoch=20,
        k_means_end_epoch=25,
        decay_rate=0.8,
    )

    return n_clusters_list


def plot_umap(umap_features_np, umap_labels, plot_dir, epoch, samples_to_track=[]):
    # Plot UMAP before clustering update
    fig = plt.figure(figsize=(16, 16))
    tmp_labels = umap_labels >= 0

    plt.scatter(
        umap_features_np[~tmp_labels, 0],
        umap_features_np[~tmp_labels, 1],
        c=[0.5, 0.5, 0.5],
        s=0.1,
        alpha=0.5,
    )

    plt.scatter(
        umap_features_np[tmp_labels, 0],
        umap_features_np[tmp_labels, 1],
        c=umap_labels[tmp_labels],
        s=0.1,
        alpha=0.5,
    )

    # Highlight and annotate the tracked samples
    if len(samples_to_track) > 0:
        for sample_idx in samples_to_track:
            x, y = umap_features_np[sample_idx, :]
            plt.scatter(x, y, c="red", s=150, edgecolors="k")  # Highlight the sample
            plt.text(
                x, y, f"Sample {sample_idx}", fontsize=24, color="black"
            )  # Annotate the sample

    # Add the number of umap_labels to the plot as title
    plt.title(f"UMAP clusters at epoch {epoch}")

    plt.colorbar()
    # output the figure
    plt.savefig(os.path.join(plot_dir, f"umap_{epoch}.png"))
    plt.close(fig)


def plot_umap_nooutlier(
    umap_features_np, umap_labels, plot_dir, epoch, samples_to_track=[], z_threshold=3
):
    # Compute the z-scores of the features for each dimension
    z_scores = np.abs(stats.zscore(umap_features_np, axis=0))

    # Filter the points by applying a z-score threshold (default = 3)
    non_outliers = (z_scores < z_threshold).all(axis=1)

    # Apply the outlier filter
    filtered_features = umap_features_np[non_outliers]
    filtered_labels = umap_labels[non_outliers]

    # Plot UMAP without outliers
    fig = plt.figure(figsize=(16, 16))
    tmp_labels = filtered_labels >= 0

    plt.scatter(
        filtered_features[~tmp_labels, 0],
        filtered_features[~tmp_labels, 1],
        c=[0.5, 0.5, 0.5],
        s=0.1,
        alpha=0.5,
    )

    plt.scatter(
        filtered_features[tmp_labels, 0],
        filtered_features[tmp_labels, 1],
        c=filtered_labels[tmp_labels],
        s=0.1,
        alpha=0.5,
    )

    # Highlight and annotate the tracked samples (if they are not outliers)
    if len(samples_to_track) > 0:
        for sample_idx in samples_to_track:
            if non_outliers[sample_idx]:  # Only track samples that are not outliers
                x, y = umap_features_np[sample_idx, :]
                plt.scatter(x, y, c="red", s=150, edgecolors="k")  # Highlight the sample
                plt.text(
                    x, y, f"Sample {sample_idx}", fontsize=24, color="black"
                )  # Annotate the sample

    # Add the number of UMAP labels to the plot as title
    plt.title(f"UMAP clusters at epoch {epoch} (outliers removed)")

    plt.colorbar()

    # output the figure
    plt.savefig(os.path.join(plot_dir, f"umap_{epoch}_nooutlier.png"))
    plt.close(fig)


def test_umap():
    import torch

    umap_features_np = np.random.rand(10000, 2)
    umap_labels = torch.tensor([0] * 5000 + [1] * 5000, device="cpu")
    print(umap_labels.shape)

    plot_umap(umap_features_np, umap_labels, plot_dir="tmp", epoch=0)


def main():
    res = test_calculate_n_clusters_3()
    print(res)


if __name__ == "__main__":
    main()
