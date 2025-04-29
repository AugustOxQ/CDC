import json
import math
import os

import numpy as np
import pandas as pd
import torch
import transformers
from numpy import ndarray
from PIL import Image
from shapely import contains
from sqlalchemy import all_
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# Suppress hugginface warnings
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get timestamp
import datetime


def calculate_average_precision(correct_positions, total_relevant):
    """Calculate Average Precision (AP) for the given ranks of relevant documents.

    correct_positions: Tensor of ranks where relevant documents were retrieved.
    total_relevant: Total number of relevant documents for the query.
    """
    if total_relevant == 0 or correct_positions.numel() == 0:
        return 0.0  # Return 0 if no relevant documents

    ap_sum = 0.0
    for i, rank in enumerate(correct_positions.sort()[0], 1):
        precision_at_rank = i / float(rank + 1)  # Correct for 1-based indexing
        ap_sum += precision_at_rank

    return ap_sum / total_relevant


def compute_metric_difference(metrics_1, metrics2, metrics_kwd2, new_kwd):
    metric_diff = {}

    for key in metrics_1:
        # Extract the metric name by removing the kwd prefix (everything after '/')
        metric_name = key.split("/")[1]

        # Find the corresponding key in metrics_kwd2
        corresponding_key_kwd2 = f"{metrics_kwd2}/{metric_name}"

        # Compute the difference between the corresponding metrics
        if key in metrics_1 and corresponding_key_kwd2 in metrics2:
            metric_diff[f"{new_kwd}/{metric_name}"] = (
                metrics_1[key] - metrics2[corresponding_key_kwd2]
            )
        else:
            print(f"Key {metric_name} not found in both dictionaries.")

    print(f"############start#########{new_kwd}#########################")
    for key, value in metric_diff.items():
        print(f"{key}: {value}")
    print(f"############end#########{new_kwd}#########################")

    return metric_diff


def absolute_rank(inds, mappings, captions_per_image):
    """_summary_

    Args:
        inds (_type_): _description_
        mappings (_type_): _description_
        captions_per_image (_type_): _description_
    """

    num_queries = inds.size(0)
    all_ranks = []

    for query_idx in range(num_queries):
        correct_indices = mappings[query_idx].tolist()

        query_inds = inds[query_idx]

        # Find ranks of correct indices
        if type(correct_indices) is int:
            # For single correct index
            correct_mask = query_inds == torch.tensor(correct_indices)
            correct_positions = correct_mask.nonzero(as_tuple=True)[-1].item()
            ranks = correct_positions + 1  # Convert to 1-based indexing
        else:
            ranks = []
            correct_mask = []
            for correct_index in correct_indices:
                # Find the position of the correct caption index in the sorted indices
                position = (query_inds == correct_index).nonzero(as_tuple=True)[-1]
                correct_mask.append(position)
                rank = position.item() + 1
                ranks.append(rank)
            assert len(ranks) == captions_per_image

        if type(ranks) is not list:
            ranks = [ranks]
        all_ranks.extend(ranks)

    return all_ranks


def calculate_metrics(inds, mappings, captions_per_image):
    """Calculate R-Precision and mAP for a set of rankings (inds) given the correct mappings.

    inds: Sorted indices for predictions.
    mappings: Correct mappings from queries (texts or images) to targets (images or texts).
    captions_per_image: Number of captions per image, used for calculating R-Precision for i2t.
    """
    num_queries = inds.size(0)
    R_precisions = []
    AP_scores = []
    all_ranks = []

    for query_idx in range(num_queries):
        correct_indices = mappings[query_idx].tolist()

        query_inds = inds[query_idx]

        # Find ranks of correct indices
        if type(correct_indices) is int:
            # For single correct index
            correct_mask = query_inds == torch.tensor(correct_indices, device=device)
            correct_positions = correct_mask.nonzero(as_tuple=True)[-1].item()
            ranks = correct_positions + 1  # Convert to 1-based indexing
        else:
            ranks = []
            correct_mask = []
            for correct_index in correct_indices:
                # Find the position of the correct caption index in the sorted indices
                position = (query_inds == correct_index).nonzero(as_tuple=True)[-1]
                correct_mask.append(position)
                rank = position.item() + 1
                ranks.append(rank)
            assert len(ranks) == captions_per_image

        if type(ranks) is not list:
            ranks = [ranks]
        all_ranks.extend(ranks)

        # Calculate AP for this query
        AP = 0
        for j, rank in enumerate(sorted(ranks), start=1):
            precision_at_j = j / rank  # type: ignore
            AP += precision_at_j
        AP /= captions_per_image
        AP_scores.append(AP)

    mean_ap = np.mean(AP_scores)
    meanR = np.mean(all_ranks)
    medR = np.median(all_ranks)

    return (meanR, medR, mean_ap)


# def encode_data(model, data_loader, tokenizer, label_embeddings: Tensor, device=device):
#     """Encode all images and captions loadable by `data_loader`"""
#     # switch to evaluate mode
#     model.eval()
#     print("Evaluating...")

#     # Lists to keep all the embeddings
#     img_embs = []
#     cap_embs = []

#     #  (as there are multiple pieces of text for each image)
#     image_to_text_map = []

#     # text_to_image_map[i] gives the corresponding image index for the ith text
#     text_to_image_map = []

#     text_index = 0
#     image_index = 0

#     with torch.no_grad():
#         for batch_id, batch in enumerate(tqdm(data_loader)):
#             image, text = batch
#             image = image.to(device)
#             captions = tokenizer(
#                 text,
#                 return_tensors="pt",
#                 padding="max_length",
#                 truncation=True,
#                 max_length=77,
#             ).to(device)

#             batch_size = image["pixel_values"].shape[0]
#             captions_per_image = data_loader.dataset.captions_per_image

#             # Update text_to_image_map and image_to_text_map for this batch
#             for batch_id in range(batch_size):
#                 # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
#                 text_indices = list(range(text_index, text_index + captions_per_image))
#                 image_to_text_map.append(text_indices)
#                 text_index += captions_per_image

#                 # Each of the next captions_per_image text captions correspond to the same image
#                 text_to_image_map += [image_index] * captions_per_image
#                 image_index += 1

#             captions = torch.flatten(captions, start_dim=0, end_dim=1)

#             img_emb, txt_emb = model.encode_img_txt(image, captions)

#             # Convert PyTorch tensors to NumPy arrays
#             img_emb_np = img_emb.cpu().numpy()
#             txt_emb_np = txt_emb.cpu().numpy()

#             best_cosine_sim = np.full(batch_size, -1.0)  # Initialize with -1
#             best_comb_emb = np.zeros((batch_size, label_embeddings.size(1)))

#             img_embs.append(img_emb)
#             cap_embs.append(txt_emb)

#     image_embeddings = torch.cat(img_embs, axis=0)  # type: ignore
#     text_embeddings = torch.cat(cap_embs, axis=0)  # type: ignore
#     text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
#     image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

#     image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
#     text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

#     return image_embeddings, text_embeddings, text_to_image_map, image_to_text_map


def evalrank_i2t(
    image_embeddings,
    text_embeddings,
    text_to_image_map,
    image_to_text_map,
    kwd: str = "",
):
    # print(image_embeddings.shape, text_embeddings.shape)
    # print(text_to_image_map.shape, image_to_text_map.shape)

    num_text = text_embeddings.shape[0]
    num_im = image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10]

    # image-to-text recall
    print("Image-to-text recall...")

    dist_matrix = text_embeddings @ image_embeddings.T  # dist_matrix[i] gives logits for ith text

    dist_matrix = dist_matrix.cpu()
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    # print(inds.shape)

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im * 100)  #

    meanR_i2t, medR_i2t, mAP_i2t = calculate_metrics(inds, image_to_text_map, captions_per_image)

    print("Done.")
    metrics = {
        f"{kwd}/i2t_R1": image_to_text_recall[0],
        f"{kwd}/i2t_R5": image_to_text_recall[1],
        f"{kwd}/i2t_R10": image_to_text_recall[2],
        f"{kwd}/i2t_meanR": meanR_i2t,
        f"{kwd}/i2t_medR": medR_i2t,
        f"{kwd}/i2t_mAP": mAP_i2t,
        f"{kwd}/i2t_rsum": sum(image_to_text_recall),
    }

    print(f"############start#########{kwd}#########################")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"############end#########{kwd}#########################")

    return metrics


def evalrank_t2i(
    image_embeddings,
    text_embeddings,
    text_to_image_map,
    image_to_text_map,
    kwd: str = "",
):
    # print(image_embeddings.shape, text_embeddings.shape)
    # print(text_to_image_map.shape, image_to_text_map.shape)

    num_text = text_embeddings.shape[0]
    num_im = image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10]

    # text-to-image recall
    print("Text-to-image recall...")

    dist_matrix = text_embeddings @ image_embeddings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)
    # print(inds.shape)

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text * 100)

    meanR_t2i, medR_t2i, mAP_t2i = calculate_metrics(inds, text_to_image_map, 1)

    print("Done.")
    metrics = {
        f"{kwd}/t2i_R1": text_to_image_recall[0],
        f"{kwd}/t2i_R5": text_to_image_recall[1],
        f"{kwd}/t2i_R10": text_to_image_recall[2],
        f"{kwd}/t2i_meanR": meanR_t2i,
        f"{kwd}/t2i_medR": medR_t2i,
        f"{kwd}/t2i_mAP": mAP_t2i,
        f"{kwd}/t2i_rsum": sum(text_to_image_recall),
    }

    print(f"############start#########{kwd}#########################")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"############end#########{kwd}#########################")

    return metrics


def compute_recall_metrics(
    best_inds_tti,
    best_inds_itt,
    text_to_image_map,
    image_to_text_map,
    k_vals,
    num_text,
    num_image,
    captions_per_image,
):
    # text-to-image recall
    best_inds_tti = best_inds_tti.to(device)  # type: ignore

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = best_inds_tti[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text * 100)

    meanR_t2i, medR_t2i, mAP_t2i = calculate_metrics(best_inds_tti, text_to_image_map, 1)

    # image-to-text recall
    best_inds_itt = best_inds_itt.to(device)  # type: ignore

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = best_inds_itt[:, :k]

        correct = torch.zeros((num_image,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_image * 100)  #

    meanR_i2t, medR_i2t, mAP_i2t = calculate_metrics(
        best_inds_itt, image_to_text_map, captions_per_image
    )

    torch.cuda.empty_cache()

    return (
        text_to_image_recall,
        meanR_t2i,
        medR_t2i,
        mAP_t2i,
        image_to_text_recall,
        meanR_i2t,
        medR_i2t,
        mAP_i2t,
    )


def evalrank_all(
    image_embeddings,
    text_embeddings,
    text_to_image_map,
    image_to_text_map,
    kwd: str = "",
):
    # print(image_embeddings.shape, text_embeddings.shape)
    # print(text_to_image_map.shape, image_to_text_map.shape)

    num_text = text_embeddings.shape[0]
    num_im = image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10]

    # Normalize embeddings
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    # text-to-image recall
    # print("Text-to-image recall...")

    dist_matrix = text_embeddings @ image_embeddings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    text_to_image_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text * 100)

    meanR_t2i, medR_t2i, mAP_t2i = calculate_metrics(inds, text_to_image_map, 1)

    # image-to-text recall
    # print("Image-to-text recall...")
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(device)

    image_to_text_recall = []

    for k in k_vals:
        # Extract top k indices only
        topk = inds[:, :k]

        correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

        #  For each image, check whether one of the 5 relevant captions was retrieved
        # Check if image matches its ith caption (for i=0..4)
        for i in range(captions_per_image):
            contains_index = torch.eq(topk, image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
            correct = torch.logical_or(correct, contains_index)

        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im * 100)  #

    meanR_i2t, medR_i2t, mAP_i2t = calculate_metrics(inds, image_to_text_map, captions_per_image)

    # print("Done.")
    metrics = {
        f"{kwd}/i2t_R1": image_to_text_recall[0],
        f"{kwd}/i2t_R5": image_to_text_recall[1],
        f"{kwd}/i2t_R10": image_to_text_recall[2],
        f"{kwd}/i2t_rsum": sum(image_to_text_recall),
        f"{kwd}/i2t_meanR": meanR_i2t,
        f"{kwd}/i2t_medR": medR_i2t,
        f"{kwd}/i2t_mAP": mAP_i2t,
        f"{kwd}/t2i_R1": text_to_image_recall[0],
        f"{kwd}/t2i_R5": text_to_image_recall[1],
        f"{kwd}/t2i_R10": text_to_image_recall[2],
        f"{kwd}/t2i_rsum": sum(text_to_image_recall),
        f"{kwd}/t2i_meanR": meanR_t2i,
        f"{kwd}/t2i_medR": medR_t2i,
        f"{kwd}/t2i_mAP": mAP_t2i,
    }

    print(f"############start-{kwd}#########################")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"############end-{kwd}#########################\n")

    return metrics


def eval_rank_oracle(
    model,
    label_embeddings,  # shape: (N_label, label_dim)
    all_img_emb,  # shape: (N_img, emb_dim)
    all_txt_emb,  # shape: (N_txt, txt_emb_dim)
    all_txt_full,  # shape: (N_txt, other_dim)  额外文本信息
    text_to_image_map,
    image_to_text_map,
    kwd: str = "",
    use_best_label: bool = True,
):
    model.eval()

    num_text = all_txt_emb.shape[0]
    num_image = all_img_emb.shape[0]
    num_labels = label_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10]

    # Moving to GPU
    all_img_emb /= all_img_emb.norm(dim=-1, keepdim=True)
    all_img_emb = all_img_emb.to(device)
    all_txt_emb = all_txt_emb.to(device)
    all_txt_full = all_txt_full.to(device)

    # BEST RANKING
    # looping over all labels
    best_inds_tti = None  # (n_text, n_image)
    best_rank_tti = torch.full((num_text,), math.inf)

    worst_inds_tti = None  # (n_text, n_image)
    worst_rank_tti = torch.full((num_text,), -1)

    best_inds_itt = None
    best_rank_itt = torch.full(
        (num_image,), math.inf
    )  # TODO: Mind this to be num_text and compute the actual abs rank instead abs mean rank

    worst_inds_itt = None
    worst_rank_itt = torch.full((num_image,), -1)

    # Add a record of which label is finally used for each text
    best_label_tti = -1 * torch.ones((num_text,))
    best_label_itt = -1 * torch.ones((num_image,))

    worst_label_tti = -1 * torch.ones((num_text,))
    worst_label_itt = -1 * torch.ones((num_image,))

    # First RANKING
    first_inds_tti = None  # (n_text, n_image)
    first_rank_tti = torch.full((num_text,), math.inf)

    first_inds_itt = None
    first_rank_itt = torch.full((num_image,), math.inf)

    # Add a record of which label is first used for each text
    first_label_tti = -1 * torch.ones((num_text,))
    first_label_itt = -1 * torch.ones((num_image,))

    print("Evaluating the best label")
    with torch.no_grad():
        for label_id in tqdm(range(-1, num_labels)):
            if label_id == -1:
                comb_embds = all_txt_emb.detach().clone()
            else:
                label_emb = label_embeddings[label_id]
                # broadcast to the same length as all_txt_emb
                label_emb = label_emb.expand(num_text, -1)
                label_emb = label_emb.to(device)
                batch_size = 256
                comb_embs = []
                for i in range(0, num_text, batch_size):
                    tmp_comb_embeds = model.module.combine(
                        all_txt_emb[i : i + batch_size],
                        all_txt_full[i : i + batch_size],
                        label_emb[i : i + batch_size],
                    ).detach()
                    comb_embs.append(tmp_comb_embeds)
                comb_embds = torch.cat(comb_embs, dim=0)
            comb_embds /= comb_embds.norm(dim=-1, keepdim=True)

            # text-to-image
            dist_matrix_tti = comb_embds @ all_img_emb.T
            dist_matrix_tti = dist_matrix_tti.cpu()
            inds_tti = torch.argsort(dist_matrix_tti, dim=1, descending=True)

            # Calculate absolute ranking
            abs_rank_tti = absolute_rank(inds_tti, text_to_image_map, 1)
            abs_rank_tti = torch.tensor(abs_rank_tti, dtype=torch.float32).cpu()

            if best_inds_tti is None:
                best_inds_tti = inds_tti.clone()
                best_rank_tti = abs_rank_tti.clone()
                first_inds_tti = inds_tti.clone()
                first_rank_tti = abs_rank_tti.clone()
                worst_inds_tti = inds_tti.clone()
                worst_rank_tti = abs_rank_tti.clone()
                update_mask = torch.ones((num_text,), dtype=torch.bool)
                update_mask_2 = torch.ones((num_text,), dtype=torch.bool)
            else:
                # For each query, if the current ranking can recall the correct image and the previous one cannot, update
                update_mask = abs_rank_tti < best_rank_tti
                if update_mask.any():
                    best_inds_tti[update_mask] = inds_tti[update_mask]
                    best_rank_tti[update_mask] = abs_rank_tti[update_mask]
                    best_label_tti[update_mask] = label_id

                update_mask_2 = abs_rank_tti > worst_rank_tti
                if update_mask_2.any():
                    worst_inds_tti[update_mask_2] = inds_tti[update_mask_2]  # type: ignore
                    worst_rank_tti[update_mask_2] = abs_rank_tti[update_mask_2]
                    worst_label_tti[update_mask_2] = label_id

                update_mask_first = first_label_tti == -1
                hybrid_mask = update_mask & update_mask_first
                if hybrid_mask.any():
                    first_inds_tti[hybrid_mask] = inds_tti[hybrid_mask]  # type: ignore
                    first_rank_tti[hybrid_mask] = abs_rank_tti[hybrid_mask]
                    first_label_tti[hybrid_mask] = label_id

            # image-to-text
            dist_matrix_itt = dist_matrix_tti.T
            dist_matrix_itt = dist_matrix_itt.cpu()
            inds_itt = torch.argsort(dist_matrix_itt, dim=1, descending=True)

            # Calculate absolute ranking
            abs_rank_itt = absolute_rank(inds_itt, image_to_text_map, captions_per_image)
            abs_rank_itt = torch.tensor(abs_rank_itt, dtype=torch.float32).cpu()

            # Sum of ranks per 5 captions, that is sum index i*5 : i*5+5 for each image
            abs_rank_itt_min = abs_rank_itt.view(-1, captions_per_image).min(dim=1).values
            abs_rank_itt_sum = abs_rank_itt.view(-1, captions_per_image).sum(dim=1)

            abs_rank_itt = abs_rank_itt_sum + abs_rank_itt_min

            if best_inds_itt is None:
                best_inds_itt = inds_itt.clone()
                best_rank_itt = abs_rank_itt.clone()
                first_inds_itt = inds_itt.clone()
                first_rank_itt = abs_rank_itt.clone()
                worst_inds_itt = inds_itt.clone()
                worst_rank_itt = abs_rank_itt.clone()
                update_mask = torch.ones((num_image,), dtype=torch.bool)
                update_mask_2 = torch.ones((num_image,), dtype=torch.bool)
            else:
                # For each query, if the current ranking can recall the correct image and the previous one cannot, update
                update_mask = abs_rank_itt < best_rank_itt
                if update_mask.any():
                    best_inds_itt[update_mask] = inds_itt[update_mask]
                    best_rank_itt[update_mask] = abs_rank_itt[update_mask]
                    best_label_itt[update_mask] = label_id

                update_mask_2 = abs_rank_itt > worst_rank_itt
                if update_mask_2.any():
                    worst_inds_itt[update_mask_2] = inds_itt[update_mask_2]  # type: ignore
                    worst_rank_itt[update_mask_2] = abs_rank_itt[update_mask_2]
                    worst_label_itt[update_mask_2] = label_id

                update_mask_first = first_label_itt == -1
                hybrid_mask = update_mask & update_mask_first
                if hybrid_mask.any():
                    first_inds_itt[hybrid_mask] = inds_itt[hybrid_mask]  # type: ignore
                    first_rank_itt[hybrid_mask] = abs_rank_itt[hybrid_mask]
                    first_label_itt[hybrid_mask] = label_id

            # Compute raw rank
            dist_matrix_raw_tti = comb_embds @ all_img_emb.T
            dist_matrix_raw_tti = dist_matrix_raw_tti.cpu()
            inds_raw_tti = torch.argsort(dist_matrix_raw_tti, dim=1, descending=True)

            dist_matrix_raw_itt = dist_matrix_raw_tti.T
            dist_matrix_raw_itt = dist_matrix_raw_itt.cpu()
            inds_raw_itt = torch.argsort(dist_matrix_raw_itt, dim=1, descending=True)

            comb_embds.cpu()
            del comb_embds

            torch.cuda.empty_cache()

    if use_best_label:
        (
            text_to_image_recall,
            meanR_t2i,
            medR_t2i,
            mAP_t2i,
            image_to_text_recall,
            meanR_i2t,
            medR_i2t,
            mAP_i2t,
        ) = compute_recall_metrics(
            best_inds_tti,
            best_inds_itt,
            text_to_image_map,
            image_to_text_map,
            k_vals,
            num_text,
            num_image,
            captions_per_image,
        )
    else:
        (
            text_to_image_recall,
            meanR_t2i,
            medR_t2i,
            mAP_t2i,
            image_to_text_recall,
            meanR_i2t,
            medR_i2t,
            mAP_i2t,
        ) = compute_recall_metrics(
            first_inds_tti,
            first_inds_itt,
            text_to_image_map,
            image_to_text_map,
            k_vals,
            num_text,
            num_image,
            captions_per_image,
        )

    # Worst label performance
    (
        text_to_image_recall_worst,
        meanR_t2i_worst,
        medR_t2i_worst,
        mAP_t2i_worst,
        image_to_text_recall_worst,
        meanR_i2t_worst,
        medR_i2t_worst,
        mAP_i2t_worst,
    ) = compute_recall_metrics(
        worst_inds_tti,
        worst_inds_itt,
        text_to_image_map,
        image_to_text_map,
        k_vals,
        num_text,
        num_image,
        captions_per_image,
    )

    metrics = {
        f"{kwd}/i2t_R1": image_to_text_recall[0],
        f"{kwd}/i2t_R5": image_to_text_recall[1],
        f"{kwd}/i2t_R10": image_to_text_recall[2],
        f"{kwd}/i2t_rsum": sum(image_to_text_recall),
        f"{kwd}/i2t_meanR": meanR_i2t,
        f"{kwd}/i2t_medR": medR_i2t,
        f"{kwd}/i2t_mAP": mAP_i2t,
        f"{kwd}/t2i_R1": text_to_image_recall[0],
        f"{kwd}/t2i_R5": text_to_image_recall[1],
        f"{kwd}/t2i_R10": text_to_image_recall[2],
        f"{kwd}/t2i_rsum": sum(text_to_image_recall),
        f"{kwd}/t2i_meanR": meanR_t2i,
        f"{kwd}/t2i_medR": medR_t2i,
        f"{kwd}/t2i_mAP": mAP_t2i,
    }

    metrics_2 = {
        f"{kwd}/i2t_R1_worst": image_to_text_recall_worst[0],
        f"{kwd}/i2t_R5_worst": image_to_text_recall_worst[1],
        f"{kwd}/i2t_R10_worst": image_to_text_recall_worst[2],
        f"{kwd}/i2t_rsum_worst": sum(image_to_text_recall_worst),
        f"{kwd}/i2t_meanR_worst": meanR_i2t_worst,
        f"{kwd}/i2t_medR_worst": medR_i2t_worst,
        f"{kwd}/i2t_mAP_worst": mAP_i2t_worst,
        f"{kwd}/t2i_R1_worst": text_to_image_recall_worst[0],
        f"{kwd}/t2i_R5_worst": text_to_image_recall_worst[1],
        f"{kwd}/t2i_R10_worst": text_to_image_recall_worst[2],
        f"{kwd}/t2i_rsum_worst": sum(text_to_image_recall_worst),
        f"{kwd}/t2i_meanR_worst": meanR_t2i_worst,
        f"{kwd}/t2i_medR_worst": medR_t2i_worst,
    }

    print(f"############start-{kwd}#########################")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"############end-{kwd}#########################\n")

    print("############start-worst#########################")
    for key, value in metrics_2.items():
        print(f"{key}: {value}")
    print("############end-worst#########################\n")

    if use_best_label:
        return (
            metrics,
            best_label_tti,
            best_label_itt,
            inds_raw_tti,
            inds_raw_itt,
        )  # TODO: add metrics two return
    else:
        return metrics, first_label_tti, first_label_itt, inds_raw_tti, inds_raw_itt


def eval_rank_oracle_check(
    model,
    label_embeddings,  # shape: (N_label, label_dim)
    all_img_emb,  # shape: (N_img, emb_dim)
    all_txt_emb,  # shape: (N_txt, txt_emb_dim)
    all_txt_full,  # shape: (N_txt, other_dim)  额外文本信息
    text_to_image_map,
    image_to_text_map,
    best_label_tti,
    best_label_itt,
):
    model.eval()

    res = {}

    # Moving to GPU
    all_img_emb /= all_img_emb.norm(dim=-1, keepdim=True)
    all_img_emb = all_img_emb.to(device)
    all_txt_emb = all_txt_emb.to(device)
    all_txt_full = all_txt_full.to(device)

    captions_per_image = image_to_text_map.shape[1]

    comb_emb_itt = all_txt_emb.clone()
    # repeat the best label for each text the number of times it is used
    best_label_itt = best_label_itt.repeat_interleave(captions_per_image)
    valid_mask_itt = best_label_itt != -1  # Those which actually use label embeddings
    valid_indices_itt = best_label_itt[
        valid_mask_itt
    ].int()  # The label indices used for each text

    selected_labels_itt = label_embeddings[valid_indices_itt].to(device)

    with torch.no_grad():
        # batch processing #TODO Add batch processing method
        comb_embs = model.module.combine(
            all_txt_emb[valid_mask_itt],
            all_txt_full[valid_mask_itt],
            selected_labels_itt,
        ).detach()

    comb_emb_itt[valid_mask_itt] = comb_embs

    all_txt_emb /= all_txt_emb.norm(dim=-1, keepdim=True)
    comb_emb_normed = comb_emb_itt / comb_emb_itt.norm(dim=-1, keepdim=True)
    # get similarity matrix
    dist_matrix_raw = all_txt_emb @ all_img_emb.T
    dist_matrix_raw = dist_matrix_raw.cpu()

    dist_matrix_itt = comb_emb_normed @ all_img_emb.T
    dist_matrix_itt = dist_matrix_itt.cpu()

    # Compare the difference between the two similarity matrices
    diff = dist_matrix_raw - dist_matrix_itt
    diff = diff.abs().mean().item()

    print(f"Sum of the absolute difference between the two similarity matrices: {diff}")

    # Compare the corresponding difference between comb_emb_normed and all_txt_emb
    diff = comb_emb_normed - all_txt_emb
    diff = diff.abs().mean().item()

    print(f"Sum of the absolute difference between comb_emb_normed and all_txt_emb: {diff}")

    return comb_emb_itt.cpu()


def eval_rank_oracle_check_per_label(
    model,
    label_embeddings,  # shape: (N_label, label_dim)
    all_img_emb,  # shape: (N_img, emb_dim)
    all_txt_emb,  # shape: (N_txt, txt_emb_dim)
    all_txt_full,  # shape: (N_txt, other_dim)  额外文本信息
    text_to_image_map,
    image_to_text_map,
    inds_raw_tti,
    inds_raw_itt,
):
    """
    Evaluate image-to-text recall (and other metrics) using only the given label embedding.

    Args:
        model: The model to evaluate.
        label_embeddings: The label embeddings to use. Shape: (N_label, label_dim).
        all_img_emb: The image embeddings to use. Shape: (N_img, emb_dim).
        all_txt_emb: The text embeddings to use. Shape: (N_txt, txt_emb_dim).
        all_txt_full: The extra text information to use. Shape: (N_txt, other_dim).
        image_to_text_map: A 2D tensor mapping each image index to its corresponding text indices.
        text_to_image_map: A 2D tensor mapping each text index to its corresponding image indices.
        best_label_tti: The best label indices to use for text-to-image.
        best_label_itt: The best label indices to use for image-to-text.
        label_emb_index: The index of the label embedding to use. Default: 0.

    Returns:
        A tuple containing the combined embeddings and the sorted indices for image-to-text retrieval.
    """
    model.eval()

    num_im = all_img_emb.shape[0]  # type: ignore
    num_text = all_txt_emb.shape[0]

    # Moving to GPU
    all_txt_emb = all_txt_emb.to(device)
    all_txt_full = all_txt_full.to(device)

    captions_per_image = image_to_text_map.shape[1]

    with torch.no_grad():
        batch_size = 1024
        comb_emb = []
        for i in range(0, num_text, batch_size):
            min(batch_size, num_text - i)
            label_emb = label_embeddings.expand(num_text, -1)
            label_emb = label_emb.to(device)
            tmp_comb_embeds = model.module.combine(
                all_txt_emb[i : i + batch_size],
                all_txt_full[i : i + batch_size],
                label_emb[i : i + batch_size],
            ).detach()
            print(model.module.combiner.print_scalar())
            comb_emb.append(tmp_comb_embeds)
        comb_emb = torch.cat(comb_emb, dim=0)

    # Normalize embeddings
    all_img_emb /= all_img_emb.norm(dim=-1, keepdim=True)
    all_img_emb = all_img_emb.to(device)
    all_txt_emb /= all_txt_emb.norm(dim=-1, keepdim=True)
    comb_emb_normed = comb_emb / comb_emb.norm(dim=-1, keepdim=True)

    # tti part

    # Get tti similarity matrix
    dist_matrix_tti = comb_emb_normed @ all_img_emb.T
    dist_matrix_tti = dist_matrix_tti.cpu()

    # Sort in descending order; first is the biggest logit
    inds_tti = torch.argsort(dist_matrix_tti, dim=1, descending=True)

    # Calculate absolute ranking
    abs_rank_tti = absolute_rank(inds_tti, text_to_image_map, 1)
    abs_rank_tti = torch.tensor(abs_rank_tti, dtype=torch.float32).cpu()

    abs_rank_tti_raw = absolute_rank(inds_raw_tti, text_to_image_map, 1)
    abs_rank_tti_raw = torch.tensor(abs_rank_tti_raw, dtype=torch.float32).cpu()

    # Check of which element the combined embedding is better than raw
    mask_tti = abs_rank_tti < abs_rank_tti_raw

    # itt part
    # Get iit similarity matrix
    dist_matrix_itt = dist_matrix_tti.T
    dist_matrix_itt = dist_matrix_itt.cpu()

    # Sort in descending order; first is the biggest logit
    inds_itt = torch.argsort(dist_matrix_itt, dim=1, descending=True)

    # Calculate absolute ranking
    abs_rank_itt = absolute_rank(inds_itt, image_to_text_map, captions_per_image)
    abs_rank_itt = torch.tensor(abs_rank_itt, dtype=torch.float32).cpu()

    # Sum of ranks per 5 captions, that is sum index i*5 : i*5+5 for each image
    abs_rank_itt_min = abs_rank_itt.view(-1, captions_per_image).min(dim=1).values
    abs_rank_itt_sum = abs_rank_itt.view(-1, captions_per_image).sum(dim=1)

    abs_rank_itt = abs_rank_itt_sum + abs_rank_itt_min

    # Calculate absolute ranking raw
    abs_rank_itt_raw = absolute_rank(inds_raw_itt, image_to_text_map, captions_per_image)
    abs_rank_itt_raw = torch.tensor(abs_rank_itt_raw, dtype=torch.float32).cpu()

    abs_rank_itt_min_raw = abs_rank_itt_raw.view(-1, captions_per_image).min(dim=1).values
    abs_rank_itt_sum_raw = abs_rank_itt_raw.view(-1, captions_per_image).sum(dim=1)

    abs_rank_itt_raw = abs_rank_itt_sum_raw + abs_rank_itt_min_raw

    # Check of which element the combined embedding is better than raw
    mask_itt = abs_rank_itt < abs_rank_itt_raw

    return (
        comb_emb.cpu(),
        inds_tti,
        mask_tti,
        inds_itt,
        mask_itt,
    )


def main(): ...


if __name__ == "__main__":
    main()
