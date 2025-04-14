import json
import os
import random
import re
import warnings
from datetime import datetime
from tabnanny import verbose
from turtle import update
from typing import List

import hydra
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from hydra import compose, initialize, initialize_config_dir, initialize_config_module
from IPython.display import clear_output, display
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sympy import count_ops, use
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

# Import local packages
from src.data.cdc_datamodule import CDC_test
from src.models.cdc import CDC
from src.models.components.clustering import Clustering, UMAP_vis
from src.utils import (
    EmbeddingManager,
    print_model_info,
)
from src.utils.evaltools import eval_rank_oracle_check_per_label
from src.utils.inference import encode_data, inference_test

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Hydra
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(config_path="configs", version_base=None)
cfg = compose(config_name="redcaps")
print(*cfg, sep="\n")


import os
import sys

import pandas as pd
import torch
from PIL import Image

pd.set_option("display.max_colwidth", None)  # Ensures full text is shown
pd.set_option("display.max_rows", 200)  # Increase max rows if needed
pd.set_option("display.max_columns", 50)  # Increase max columns if needed


def plot_umap(umap_features_np, umap_labels, cluster_centers, representatives):
    # Plot UMAP before clustering update
    fig = plt.figure(figsize=(16, 16))
    tmp_labels = umap_labels >= 0

    if umap_features_np is not None:
        plt.scatter(
            umap_features_np[~tmp_labels, 0],
            umap_features_np[~tmp_labels, 1],
            c=[0.5, 0.5, 0.5],
            s=0.2,
            alpha=0.5,
        )

        plt.scatter(
            umap_features_np[tmp_labels, 0],
            umap_features_np[tmp_labels, 1],
            c=umap_labels[tmp_labels],
            s=0.2,
            alpha=0.5,
        )

    if cluster_centers is not None:
        plt.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            c="black",
            s=100,
            marker="x",
            label="Cluster Centers",
        )

    if representatives is not None:
        plt.scatter(
            representatives[:, 0],
            representatives[:, 1],
            c="red",
            s=100,
            marker="o",
            label="Representatives",
        )

    # Add the number of umap_labels to the plot as title
    plt.title("UMAP with cluster_centers")
    plt.colorbar()
    return fig


# Set seed
seed = cfg.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Define the parent folder
parent_folder = "res"

# res_path = "/project/Deep-Clustering/res/20250408_011226_flickr30k"
# res_path = "/project/Deep-Clustering/res/20250408_201053_flickr30k"
res_path = "/project/Deep-Clustering/res/20250408_123658_redcaps-preextracted"

if res_path is None:
    print("No path provided. Searching for the latest experiment...")
    # Get a list of all subdirectories inside the parent folder
    subfolders = [
        os.path.join(parent_folder, d)
        for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d))
    ]

    # Sort subfolders by modification time (newest first)
    res_path = max(subfolders, key=os.path.getmtime) if subfolders else None

print(f"Using results from: {res_path}")


use_best_label = True


def main():
    # Initialize Model
    model = CDC(
        clip_trainable=False,
        d_model=cfg.model.d_model,
        nhead=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        label_dim=cfg.model.label_dim,
    )
    model = nn.DataParallel(model)
    # load model
    model.load_state_dict(torch.load(f"{res_path}/final_model.pth"))
    model.to(device)

    clustering = Clustering()
    umap_vis = UMAP_vis()

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ann_path = cfg.dataset.test_path
    ann = json.load(open(ann_path))

    if len(ann) > 5000:
        ratio = 5000 / len(ann)
    else:
        ratio = 1

    # ratio = 1

    test_dataset = CDC_test(
        annotation_path=cfg.dataset.test_path,
        image_path=cfg.dataset.img_path_test,
        processor=processor,
        ratio=ratio,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    unique_embeddings = torch.load(f"{res_path}/unique_embeddings.pt")

    embedding_manager = EmbeddingManager(
        ann,
        embedding_dim=cfg.model.label_dim,
        chunk_size=cfg.train.batch_size,
        embeddings_dir=f"{res_path}/init/",
        load_existing=True,
        sample_ids_list=None,
    )
    all_embeddings = embedding_manager.get_all_embeddings()
    sample_ids, label_embedding = embedding_manager.get_all_embeddings()

    umap_features = umap_vis.learn_umap(label_embedding, n_components=2)

    print("##########Performing Clustering##########")
    umap_labels, _ = clustering.get_hdbscan(umap_features, n_clusters=0, method="leaf")
    umap_features_cluster = umap_vis.predict_umap(unique_embeddings.cpu().numpy())

    umap_features_np = umap_features.cpu().numpy()
    umap_labels_np = umap_labels.cpu().numpy()
    umap_features_cluster_np = umap_features_cluster.cpu().numpy()
    store_path_0 = "/project/Deep-Clustering/ckpt/tmp0"
    # Check if the directory exists, if not, create it
    if not os.path.exists(store_path_0):
        os.makedirs(store_path_0)

    # Save the UMAP features and labels
    np.save(os.path.join(store_path_0, "umap_features.npy"), umap_features_np)
    np.save(os.path.join(store_path_0, "umap_labels.npy"), umap_labels_np)
    np.save(
        os.path.join(store_path_0, "umap_features_cluster.npy"),
        umap_features_cluster_np,
    )

    kmeans = KMeans(n_clusters=min(50, umap_features_cluster_np.shape[0])).fit(
        umap_features_cluster_np
    )
    centroids = kmeans.cluster_centers_
    # Find closest real embedding to each centroid

    indices = np.argmin(cdist(centroids, umap_features_cluster_np), axis=1)
    representatives = umap_features_cluster_np[indices]

    unique_embeddings = torch.from_numpy(representatives)

    print("##########Testing test dataset##########")
    (
        img_emb,
        txt_emb,
        txt_full,
        text_to_image_map,
        image_to_text_map,
        inds_raw_tti,
        inds_raw_itt,
    ) = encode_data(
        model,
        processor,
        test_dataloader,
        device,
    )

    # Go with itt experiments through all labels for a single image
    inds_tti_all = []
    mask_tti_all = []
    inds_itt_all = []
    mask_itt_all = []
    store_path = "/project/Deep-Clustering/ckpt/tmp"
    # Check if the directory exists, if not, create it
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    store_path_2 = "/project/Deep-Clustering/ckpt/tmp2"
    # Check if the directory exists, if not, create it
    if not os.path.exists(store_path_2):
        os.makedirs(store_path_2)

    save_id = 0
    for idx, selected_label in enumerate(tqdm(unique_embeddings)):
        (
            _,
            ints_tti,
            mask_tti,
            inds_itt,
            mask_itt,
        ) = eval_rank_oracle_check_per_label(
            model,
            selected_label,
            img_emb,
            txt_emb,
            txt_full,
            text_to_image_map,
            image_to_text_map,
            inds_raw_tti=inds_raw_tti,
            inds_raw_itt=inds_raw_itt,
        )
        inds_tti_all.append(ints_tti.detach().cpu().numpy().astype(np.uint16))
        mask_tti_all.append(mask_tti.detach().cpu().numpy().astype(np.uint16))
        inds_itt_all.append(inds_itt.detach().cpu().numpy().astype(np.uint16))
        mask_itt_all.append(mask_itt.detach().cpu().numpy().astype(np.uint16))

        # comb_emb = comb_emb.detach().cpu().numpy()
        # torch.save(
        #     comb_emb,
        #     f"{store_path_2}/comb_emb_{idx}.pt",
        # )
        # Save the buffer every 10 iterations
        if (idx + 1) % 10 == 0 or idx == len(unique_embeddings) - 1:
            np.save(
                f"{store_path}/inds_tti_all_{save_id}.npy",
                np.array(inds_tti_all).astype(np.uint16),
            )
            np.save(
                f"{store_path}/mask_tti_all_{save_id}.npy",
                np.array(mask_tti_all).astype(np.uint16),
            )
            np.save(
                f"{store_path}/inds_itt_all_{save_id}.npy",
                np.array(inds_itt_all).astype(np.uint16),
            )
            np.save(
                f"{store_path}/mask_itt_all_{save_id}.npy",
                np.array(mask_itt_all).astype(np.uint16),
            )
            save_id += 1
            # clear the buffer
            inds_tti_all.clear()
            mask_tti_all.clear()
            inds_itt_all.clear()
            mask_itt_all.clear()

    """
    1. inds_itt is the indices of the itt inds using combined embeddings
    2. mask_itt is the mask of the image-shape that indicate which image improved by using the selected label.
    """


if __name__ == "__main__":
    main()
