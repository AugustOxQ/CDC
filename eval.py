import json
import os
import random
import warnings
from cProfile import label
from datetime import datetime
from turtle import update

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
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
    print_model_info,
)
from src.utils.evaltools import eval_rank_oracle_check_per_label
from src.utils.inference import (
    inference_test,
)

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg: DictConfig, **kwargs):
    res_path = kwargs.get("res_path", None)
    use_best_label = True

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

    # Print model summary
    # print_model_info(model)

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ann_path = cfg.dataset.test_path
    ann = json.load(open(ann_path))

    if len(ann) > 5000:
        ratio = 5000 / len(ann)
    else:
        ratio = 1

    del ann_path, ann

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

    print("##########Testing test dataset##########")
    unique_embeddings = torch.load(f"{res_path}/unique_embeddings.pt")
    (
        img_emb,
        txt_emb,
        txt_full,
        text_to_image_map,
        image_to_text_map,
        best_label_tti,
        best_label_itt,
        inds_raw_tti,
        inds_raw_itt,
    ) = inference_test(
        model,
        processor,
        test_dataloader,
        unique_embeddings,
        -1,
        device,
        inspect_labels=True,
        use_best_label=use_best_label,
    )

    # First use img_emb + txt_emb to create a cat_emb and compute umap
    cat_emb = torch.cat((img_emb, txt_emb), dim=0)

    # save_path
    umap_path = "/project/Deep-Clustering/notebooks/inspect_testset_label/UMAP_plot"
    # Check if the path exists, if not create it
    if not os.path.exists(umap_path):
        os.makedirs(umap_path)
    umap_vis = UMAP_vis()
    umap_features = umap_vis.learn_umap(cat_emb, n_components=2)
    umap_features_raw_image = umap_features[: img_emb.shape[0], :]
    umap_features_raw_text = umap_features[img_emb.shape[0] :, :]

    # Compute original umap
    fig = plt.figure(figsize=(10, 10))

    # Plot image embeddings
    plt.scatter(
        umap_features[: img_emb.shape[0], 0],
        umap_features[: img_emb.shape[0], 1],
        s=5,
        alpha=1,
        label="Image Embeddings",
    )

    # Plot text embeddings
    plt.scatter(
        umap_features[img_emb.shape[0] : img_emb.shape[0] + txt_emb.shape[0], 0],
        umap_features[img_emb.shape[0] : img_emb.shape[0] + txt_emb.shape[0], 1],
        s=5,
        alpha=1,
        label="Text Embeddings",
    )
    # # Plot combined embeddings
    # plt.scatter(
    #     umap_features[img_emb.shape[0] + txt_emb.shape[0] :, 0],
    #     umap_features[img_emb.shape[0] + txt_emb.shape[0] :, 1],
    #     s=5,
    #     alpha=1,
    #     label="Combined Embeddings",
    # )
    # plt.legend()

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    str_tag = "raw"
    plt.savefig(os.path.join(umap_path, f"umap_{str_tag}.png"))
    plt.close()

    # Find unique values in best_label
    unique_values_label_itt, counts_label_itt = torch.unique(best_label_itt, return_counts=True)

    # Sort counts_label_itt
    sorted_indices_label_itt = torch.argsort(counts_label_itt, descending=True)

    inds_collection = []
    all_selected_label_indices_collection = []

    N = len(sorted_indices_label_itt)
    sorted_indices_label_collection = unique_values_label_itt[sorted_indices_label_itt[:N]]

    print(f"##########Evaluating top {N} labels##########")

    for idx, selected_label_indices in tqdm(enumerate(sorted_indices_label_collection)):

        # Get all indices where this label is used
        all_selected_label_indices = torch.where(best_label_itt == selected_label_indices)[0]

        comb_emb_itt, inds_itt, mask_itt, mask_itt_expand = eval_rank_oracle_check_per_label(
            model,
            unique_embeddings,
            img_emb,
            txt_emb,
            txt_full,
            text_to_image_map,
            image_to_text_map,
            label_emb_index=int(selected_label_indices),
            inds_raw_itt=inds_raw_itt,
        )

        inds_collection.append(inds_itt)
        all_selected_label_indices_collection.append(all_selected_label_indices)

        # Plot UMAP for all points
        fig = plt.figure(figsize=(10, 10))

        comb_emb_itt = comb_emb_itt.detach().cpu().numpy()
        umap_features_new = umap_vis.predict_umap(comb_emb_itt)

        # Compute original umap
        fig = plt.figure(figsize=(10, 10))
        # Plot image embeddings
        plt.scatter(
            umap_features_raw_image[:, 0],
            umap_features_raw_image[:, 1],
            s=5,
            alpha=1,
            label="Image Embeddings",
        )

        # Plot text embeddings
        plt.scatter(
            umap_features_raw_text[:, 0],
            umap_features_raw_text[:, 1],
            s=5,
            alpha=1,
            label="Text Embeddings",
        )

        # Plot combined embeddings
        plt.scatter(
            umap_features_new[:, 0],
            umap_features_new[:, 1],
            s=5,
            alpha=1,
            label="Combined Embeddings",
        )
        plt.legend()

        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        str_tag = "best" if use_best_label else "first"
        plt.savefig(os.path.join(umap_path, f"umap_{str_tag}_label_{idx}_all.png"))
        plt.close()

        # Plot UMAP for improved points only
        fig = plt.figure(figsize=(10, 10))

        # Plot image embeddings
        plt.scatter(
            umap_features_raw_image[mask_itt, 0],
            umap_features_raw_image[mask_itt, 1],
            s=5,
            alpha=1,
            label="Image Embeddings (Improved)",
        )

        # Plot text embeddings
        plt.scatter(
            umap_features_raw_text[mask_itt_expand, 0],
            umap_features_raw_text[mask_itt_expand, 1],
            s=5,
            alpha=1,
            label="Text Embeddings (Improved)",
        )

        # Plot combined embeddings
        plt.scatter(
            umap_features_new[mask_itt_expand, 0],
            umap_features_new[mask_itt_expand, 1],
            s=5,
            alpha=1,
            label="Combined Embeddings (Improved)",
        )

        # Plot image embeddings
        plt.scatter(
            umap_features_raw_image[~mask_itt, 0],
            umap_features_raw_image[~mask_itt, 1],
            s=5,
            alpha=0.05,
            color="blue",
        )

        # Plot text embeddings
        plt.scatter(
            umap_features_raw_text[~mask_itt_expand, 0],
            umap_features_raw_text[~mask_itt_expand, 1],
            s=5,
            alpha=0.05,
            color="orange",
        )

        # Plot combined embeddings
        plt.scatter(
            umap_features_new[~mask_itt_expand, 0],
            umap_features_new[~mask_itt_expand, 1],
            s=5,
            alpha=0.05,
            color="green",
        )

        plt.legend()

        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        str_tag = "best" if use_best_label else "first"
        plt.savefig(os.path.join(umap_path, f"umap_{str_tag}_label_{idx}_improved_only.png"))
        plt.close()

    umap_vis.close_cluster()

    res = {}
    res["sorted_indices_label_collection"] = sorted_indices_label_collection
    res["all_selected_label_indices_collection"] = all_selected_label_indices_collection
    res["inds_collection"] = inds_collection

    res["inds_raw_tti"] = inds_raw_tti
    res["inds_raw_itt"] = inds_raw_itt

    # save results
    torch.save(
        res,
        f"/project/Deep-Clustering/notebooks/inspect_testset_label/eval_results_top_{N}.pt",
    )

    # Clean cuda cache
    del (model,)
    torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="redcaps", version_base=None)
def main(cfg):
    # Set seed
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Define the parent folder
    parent_folder = "res"

    res_path = "/project/Deep-Clustering/res/20250219_043513_mscoco-preextracted"

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

    # Run main function
    run(cfg=cfg, res_path=res_path)


if __name__ == "__main__":
    main()
