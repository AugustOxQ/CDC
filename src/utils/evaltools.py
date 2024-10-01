import json
import os

import numpy as np
import pandas as pd
import torch
import transformers
from numpy import ndarray
from PIL import Image
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
        if type(correct_indices) == int:
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

        if type(ranks) != list:
            ranks = [ranks]
        all_ranks.extend(ranks)

        # Calculate AP for this query
        AP = 0
        for j, rank in enumerate(sorted(ranks), start=1):
            precision_at_j = j / rank
            AP += precision_at_j
        AP /= captions_per_image
        AP_scores.append(AP)

    mean_ap = np.mean(AP_scores)
    meanR = np.mean(all_ranks)
    medR = np.median(all_ranks)

    return (meanR, medR, mean_ap)


def encode_data(model, data_loader, tokenizer, label_embeddings: Tensor, device=device):
    """Encode all images and captions loadable by `data_loader`"""
    # switch to evaluate mode
    model.eval()
    print("Evaluating...")

    # Lists to keep all the embeddings
    img_embs = []
    cap_embs = []

    #  (as there are multiple pieces of text for each image)
    image_to_text_map = []

    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    text_index = 0
    image_index = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(data_loader)):
            image, text = batch
            image = image.to(device)
            captions = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            batch_size = image["pixel_values"].shape[0]
            captions_per_image = 5

            # Update text_to_image_map and image_to_text_map for this batch
            for batch_id in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            captions = torch.flatten(captions, start_dim=0, end_dim=1)

            img_emb, txt_emb = model.encode_img_txt(image, captions)

            # Convert PyTorch tensors to NumPy arrays
            img_emb_np = img_emb.cpu().numpy()
            txt_emb_np = txt_emb.cpu().numpy()

            best_cosine_sim = np.full(batch_size, -1.0)  # Initialize with -1
            best_comb_emb = np.zeros((batch_size, label_embeddings.size(1)))

            img_embs.append(img_emb)
            cap_embs.append(txt_emb)

    image_embeddings = torch.cat(img_embs, axis=0)
    text_embeddings = torch.cat(cap_embs, axis=0)
    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    return image_embeddings, text_embeddings, text_to_image_map, image_to_text_map


def evalrank(
    image_embeddings, text_embeddings, text_to_image_map, image_to_text_map, kwd: str = ""
):
    print(image_embeddings.shape, text_embeddings.shape)
    print(text_to_image_map.shape, image_to_text_map.shape)

    num_text = text_embeddings.shape[0]
    num_im = image_embeddings.shape[0]
    captions_per_image = image_to_text_map.shape[1]
    k_vals = [1, 5, 10]
    print(
        f"Number of images: {num_im}, Number of texts: {num_text}, Captions per image: {captions_per_image}"
    )

    # # text-to-image recall
    # print("Text-to-image recall...")

    dist_matrix = text_embeddings @ image_embeddings.T  # dist_matrix[i] gives logits for ith text

    # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
    #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
    dist_matrix = dist_matrix.cpu()

    # # Sort in descending order; first is the biggest logit
    # inds = torch.argsort(dist_matrix, dim=1, descending=True)
    # inds = inds.to(device)
    # # print(inds.shape)

    # text_to_image_recall = []

    # for k in k_vals:
    #     # Extract top k indices only
    #     topk = inds[:, :k]

    #     # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
    #     correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)

    #     num_correct = correct.sum().item()
    #     text_to_image_recall.append(num_correct / num_text * 100)

    # meanR_t2i, medR_t2i, mAP_t2i = calculate_metrics(inds, text_to_image_map, 1)

    # image-to-text recall
    print("Image-to-text recall...")
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
        # f"{kwd}/text_to_image_recall_1": text_to_image_recall[0],
        # f"{kwd}/text_to_image_recall_5": text_to_image_recall[1],
        # f"{kwd}/text_to_image_recall_10": text_to_image_recall[2],
        # f"{kwd}/meanR_t2i": meanR_t2i,
        # f"{kwd}/medR_t2i": medR_t2i,
        # f"{kwd}/mAP_t2i": mAP_t2i,
        # f"{kwd}/text_to_image_rsum": sum(text_to_image_recall),
        f"{kwd}/i2t_rsum": sum(image_to_text_recall),
        # f"{kwd}/r_sum": sum(text_to_image_recall) + sum(image_to_text_recall),
    }

    print(f"#####################{kwd}#########################")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return metrics


def main():
    pass


if __name__ == "__main__":
    main()
