import os
import time
import warnings

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import all_
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import local packages
from src.data.cdc_datamodule import FeatureExtractionDataset
from src.utils import (
    compute_metric_difference,
    eval_rank_oracle,
    eval_rank_oracle_check,
    eval_rank_oracle_check_per_label,
    evalrank_all,
)

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_ranks(similarity_matrix):
    """
    Compute the rank of the diagonal elements in the sorted similarity matrix.
    """
    ranks = []
    for i in range(similarity_matrix.shape[0]):
        row = similarity_matrix[i]
        sorted_indices = np.argsort(row)[::-1]  # Descending order (higher sim first)
        rank = np.where(sorted_indices == i)[0][0] + 1  # Rank is 1-based
        ranks.append(rank)
    return np.mean(ranks)  # Compute mean rank per batch


def random_sample_with_replacement(label_embedding):
    """Randomly sample label embeddings with replacement.

    The function takes a tensor of label embeddings and returns a tensor of the same shape
    where each element is randomly sampled from the input tensor with replacement. The
    function ensures that the sampled index is not the same as the original index.

    Args:
        label_embedding (Tensor): Tensor of label embeddings. Shape (n_labels, embedding_dim).

    Returns:
        sampled_label_embedding (Tensor): Tensor of sampled label embeddings. Shape (n_labels, embedding_dim).
    """

    size = label_embedding.size(0)
    random_indices = torch.randint(0, size, (size,))

    # Ensure that sampled index is not the same as the original
    for i in range(size):
        while random_indices[i] == i:
            random_indices[i] = torch.randint(0, size, (1,))

    return label_embedding[random_indices]


def sample_label_embeddings(label_embeddings):
    """Sample new label embeddings from the set of unique label embeddings such that no sampled
    embedding is the same as the original at the same index.

    :param label_embeddings: A tensor of shape (batch_size, embedding_dim) representing the label
        embeddings.
    :return: A tensor of sampled label embeddings with the same shape, ensuring no embedding is at
        its original index.
    """
    batch_size = label_embeddings.size(0)

    batch_size = label_embeddings.size(0)

    # Find unique label embeddings
    unique_label_embeddings = torch.unique(label_embeddings, dim=0)
    num_unique = unique_label_embeddings.size(0)

    if num_unique == 1:
        return label_embeddings

    # Initialize the new embeddings tensor
    sampled_label_embeddings = torch.empty_like(label_embeddings)

    # For each label_embedding[i], sample a different embedding from unique_label_embeddings
    for i in range(batch_size):
        # Get the current label embedding
        original_embedding = label_embeddings[i]

        # Find all unique embeddings that are not equal to the original embedding
        available_choices = unique_label_embeddings[
            ~torch.all(unique_label_embeddings == original_embedding, dim=1)
        ]

        # Sample one of the available choices
        sampled_label_embeddings[i] = available_choices[
            torch.randint(0, available_choices.size(0), (1,))
        ]

    return sampled_label_embeddings


def replace_with_most_different(label_embeddings):
    """Replace each label embedding with the one that differs the most from it, based on cosine
    distance.

    :param label_embeddings: Tensor of shape [batch, 512]
    :return: Tensor of shape [batch, 512] with replaced embeddings
    """
    batch_size = label_embeddings.size(0)

    # Normalize the embeddings to compute cosine similarity
    normalized_embeddings = F.normalize(label_embeddings, p=2, dim=1)

    # Compute pairwise cosine similarity
    cosine_sim_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)

    # Convert cosine similarity to cosine distance (1 - similarity)
    cosine_dist_matrix = 1 - cosine_sim_matrix

    # For each embedding, find the index of the embedding with the maximum distance
    max_dist_indices = torch.argmax(cosine_dist_matrix, dim=1)

    # Replace each label embedding with the one that differs the most (max distance)
    new_label_embeddings = label_embeddings[max_dist_indices]

    return new_label_embeddings


def compute_recall_at_k(similarities, k):
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    correct_at_k = np.sum(top_k_indices == np.arange(similarities.shape[0])[:, None])
    return correct_at_k / similarities.shape[0]


def extract_and_store_features(
    annotation_path,
    image_path,
    feature_manager,
    batch_size,
    model,
    processor,
    device,
    ratio=0.1,
):
    dataset = FeatureExtractionDataset(annotation_path, image_path, processor, ratio=ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    sample_ids_list = []
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(dataloader)):
            images, raw_texts, sample_ids = batch

            image_input = images.to(device)
            text_input = processor(
                text=raw_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            with torch.no_grad():
                img_emb, txt_emb, _, txt_full = model.module.encode_img_txt(
                    image_input, text_input
                )
                img_emb, txt_emb, txt_full = (
                    img_emb.cpu().numpy(),
                    txt_emb.cpu().numpy(),
                    txt_full.cpu().numpy(),
                )

            feature_manager.add_features_chunk(batch_id, img_emb, txt_emb, txt_full, sample_ids)

            sample_ids_list.extend(sample_ids)

        return sample_ids_list


# TODO: Change this to also ranking-based as in inference_test
@torch.no_grad()
def inference_train(model, dataloader, device, epoch=0, Ks=[1, 5, 10], max_batches=25):
    # Read embeddings directly from the dataloader, compare with other mebeddings from the same batch
    model.eval()
    total_rank_raw = 0
    total_rank_comb = 0
    total_rank_comb_shuffled = 0
    total_samples = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(dataloader)):
            # Limit the number of batches to process
            if batch_id >= max_batches:
                print(f"Epoch {epoch}: Stopping inference after {max_batches} batches")
                break

            img_emb, txt_emb_cls, txt_emb, label_embedding, sample_id = batch
            img_emb, txt_emb_cls, txt_emb, label_embedding = (
                img_emb.squeeze(0),
                txt_emb_cls.squeeze(0),
                txt_emb.squeeze(0),
                label_embedding.squeeze(0),
            )

            img_emb = img_emb.to(device)
            txt_emb = txt_emb.to(device)
            txt_emb_cls = txt_emb_cls.to(device)
            label_embedding = label_embedding.to(device)

            # Select the most different label embeddings
            label_embedding_shuffled = replace_with_most_different(label_embedding)

            # Combine embeddings
            comb_emb = model.module.combine(txt_emb_cls, txt_emb, label_embedding)

            # Combine embeddings (shuffled)
            comb_emb_shuffled = model.module.combine(
                txt_emb_cls, txt_emb, label_embedding_shuffled
            )

            # Move to CPU for ranking calculations
            img_emb_cpu = img_emb.cpu().numpy()
            txt_emb_cls_cpu = txt_emb_cls.cpu().numpy()
            comb_emb_cpu = comb_emb.cpu().numpy()
            comb_emb_shuffled_cpu = comb_emb_shuffled.cpu().numpy()

            # Compute full similarity matrices
            sim_raw = cosine_similarity(img_emb_cpu, txt_emb_cls_cpu)  # img_emb vs txt_emb_cls
            sim_comb = cosine_similarity(img_emb_cpu, comb_emb_cpu)  # img_emb vs comb_emb
            sim_comb_shuffled = cosine_similarity(
                img_emb_cpu, comb_emb_shuffled_cpu
            )  # img_emb vs comb_emb_shuffled

            # Compute mean rank per batch
            avg_rank_raw = compute_ranks(sim_raw)
            avg_rank_comb = compute_ranks(sim_comb)
            avg_rank_comb_shuffled = compute_ranks(sim_comb_shuffled)

            total_rank_raw += avg_rank_raw
            total_rank_comb += avg_rank_comb
            total_rank_comb_shuffled += avg_rank_comb_shuffled
            total_samples += 1  # Track number of processed batches

            # Cleanup
            del (
                img_emb,
                txt_emb,
                label_embedding,
                comb_emb,
                comb_emb_shuffled,
                label_embedding_shuffled,
                img_emb_cpu,
                txt_emb_cls_cpu,
                comb_emb_cpu,
                comb_emb_shuffled_cpu,
            )
            torch.cuda.empty_cache()

    # Compute overall mean rank
    mean_rank_raw = total_rank_raw / total_samples
    mean_rank_comb = total_rank_comb / total_samples
    mean_rank_comb_shuffled = total_rank_comb_shuffled / total_samples

    print(
        f"Epoch {epoch}: Mean Rank - Raw: {mean_rank_raw:.2f}, Combined: {mean_rank_comb:.2f}, Shuffled: {mean_rank_comb_shuffled:.2f}"
    )

    return {
        "val/epoch": epoch,
        "val/mean_rank_raw": mean_rank_raw,
        "val/mean_rank_comb": mean_rank_comb,
        "val/mean_rank_comb_shuffled": mean_rank_comb_shuffled,
    }


@torch.no_grad()
def inference_test(
    model,
    processor,
    dataloader,
    label_embeddings,
    epoch,
    device,
    inspect_labels=False,
    use_best_label=False,
):
    # Load unique label embeddings up to 300
    # label_embeddings = label_embeddings[:300]
    all_img_emb = []
    all_txt_emb = []
    all_txt_full = []

    #  (as there are multiple pieces of text for each image)
    image_to_text_map = []
    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    text_index = 0
    image_index = 0

    # Accumulate embeddings for recall calculation
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            image, raw_text = batch
            image_input = image.to(device)
            batch_size = image_input["pixel_values"].size(0)
            raw_text_list = []
            batch_size, captions_per_image = (
                image["pixel_values"].shape[0],
                dataloader.dataset.captions_per_image,
            )

            # Flatten raw_text
            for b in range(batch_size):
                if captions_per_image == 1:
                    raw_text_list.append(raw_text[b])
                else:
                    for i in range(captions_per_image):
                        raw_text_list.append(raw_text[i][b])
            raw_text = raw_text_list

            # Tokenize raw_text
            text_input = processor(
                text=raw_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            # Update text_to_image_map and image_to_text_map for this batch
            for _ in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            img_emb, txt_emb, _, txt_full = model.module.encode_img_txt(image_input, text_input)

            all_img_emb.append(img_emb.cpu())
            all_txt_emb.append(txt_emb.cpu())
            all_txt_full.append(txt_full.cpu())

            del img_emb, txt_emb, image_input, text_input, txt_full

            torch.cuda.empty_cache()

    # Concate, normalize, and transform embeddings
    all_img_emb = torch.cat(all_img_emb, axis=0)  # type: ignore
    all_txt_emb = torch.cat(all_txt_emb, axis=0)  # type: ignore
    all_txt_full = torch.cat(all_txt_full, axis=0)  # type: ignore

    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

    metrics_oracle, best_label_tti, best_label_itt, inds_raw_tti, inds_raw_itt = eval_rank_oracle(
        model,
        label_embeddings,  # shape: (N_label, label_dim)
        all_img_emb,  # shape: (N_img, emb_dim)
        all_txt_emb,  # shape: (N_txt, txt_emb_dim)
        all_txt_full,  # shape: (N_txt, other_dim)  额外文本信息
        text_to_image_map,
        image_to_text_map,
        "oracle",
        use_best_label,
    )

    metrics_raw = evalrank_all(
        all_img_emb, all_txt_emb, text_to_image_map, image_to_text_map, "raw"
    )

    # Difference between two dictionaries
    metrics_diff = compute_metric_difference(metrics_oracle, metrics_raw, "raw", "diff")

    metrics_total = {
        "test/epoch": epoch,
        **metrics_oracle,
        **metrics_raw,
        **metrics_diff,
    }

    if not inspect_labels:
        return metrics_total
    else:
        str_tag = "best_label" if use_best_label else "first_label"
        # First visualize the labels with the frequency of their occurrence
        visualize_labels(best_label_tti, f"tti_{str_tag}")
        visualize_labels(best_label_itt, f"itt_{str_tag}")

        # # Combine embeddings
        # comb_emb_itt = eval_rank_oracle_check(
        #     model,
        #     label_embeddings,
        #     all_img_emb,
        #     all_txt_emb,
        #     all_txt_full,
        #     image_to_text_map,
        #     text_to_image_map,
        #     best_label_tti,
        #     best_label_itt,
        # )

        return (
            all_img_emb,
            all_txt_emb,
            all_txt_full,
            text_to_image_map,
            image_to_text_map,
            best_label_tti,
            best_label_itt,
            inds_raw_tti,
            inds_raw_itt,
        )


@torch.no_grad()
def encode_data(
    model,
    processor,
    dataloader,
    device,
):
    # Load unique label embeddings up to 300
    # label_embeddings = label_embeddings[:300]
    all_img_emb = []
    all_txt_emb = []
    all_txt_full = []

    #  (as there are multiple pieces of text for each image)
    image_to_text_map = []
    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    text_index = 0
    image_index = 0

    # Accumulate embeddings for recall calculation
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            image, raw_text = batch
            image_input = image.to(device)
            batch_size = image_input["pixel_values"].size(0)
            raw_text_list = []
            batch_size, captions_per_image = (
                image["pixel_values"].shape[0],
                dataloader.dataset.captions_per_image,
            )

            # Flatten raw_text
            for b in range(batch_size):
                if captions_per_image == 1:
                    raw_text_list.append(raw_text[b])
                else:
                    for i in range(captions_per_image):
                        raw_text_list.append(raw_text[i][b])
            raw_text = raw_text_list

            # Tokenize raw_text
            text_input = processor(
                text=raw_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            # Update text_to_image_map and image_to_text_map for this batch
            for _ in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            img_emb, txt_emb, _, txt_full = model.module.encode_img_txt(image_input, text_input)

            all_img_emb.append(img_emb.cpu())
            all_txt_emb.append(txt_emb.cpu())
            all_txt_full.append(txt_full.cpu())

            del img_emb, txt_emb, image_input, text_input, txt_full

            torch.cuda.empty_cache()

    # Concate, normalize, and transform embeddings
    all_img_emb = torch.cat(all_img_emb, axis=0)  # type: ignore
    all_txt_emb = torch.cat(all_txt_emb, axis=0)  # type: ignore
    all_txt_full = torch.cat(all_txt_full, axis=0)  # type: ignore

    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

    all_img_emb_normed = F.normalize(all_img_emb, p=2, dim=1).cpu()
    all_txt_emb_normed = F.normalize(all_txt_emb, p=2, dim=1).cpu()

    dist_matrix_raw = all_img_emb_normed @ all_txt_emb_normed.T
    inds_raw_itt = torch.argsort(dist_matrix_raw, dim=1, descending=True)
    inds_raw_tti = torch.argsort(dist_matrix_raw.T, dim=1, descending=True)

    return (
        all_img_emb,
        all_txt_emb,
        all_txt_full,
        text_to_image_map,
        image_to_text_map,
        inds_raw_tti,
        inds_raw_itt,
    )


def visualize_labels(best_label_index, tag=""):
    # Get unique values and their counts
    unique_vals, counts = torch.unique(best_label_index, return_counts=True)

    # Sort by count in descending order
    sorted_indices = torch.argsort(counts, descending=True)
    top_n = 15  # Number of top unique values to keep
    top_unique_vals = unique_vals[sorted_indices][:top_n]
    top_counts = counts[sorted_indices][:top_n]

    # Convert to numpy for plotting
    top_unique_vals_np = top_unique_vals.numpy()
    top_counts_np = top_counts.numpy()

    # Plot the distribution of top N values
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top_unique_vals_np)), top_counts_np, width=0.8, alpha=0.7)
    plt.xticks(range(len(top_unique_vals_np)), labels=top_unique_vals_np)
    plt.xlabel("Top Unique Values")
    plt.ylabel("Counts")
    plt.title(f"Top {top_n} Most Frequent Unique Values in {tag}")

    # Save the plot
    plt.show()
    plt.savefig(f"plot/plots_label_count_{tag}.png")
    plt.close()


@hydra.main(config_path="configs", config_name="coco", version_base=None)
def test(cfg: DictConfig):
    from src.models.cdc import CDC

    model = CDC()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

    from src.data.cdc_datamodule import CDC_test

    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    test_dataset = CDC_test(
        annotation_path=cfg.dataset.test_path,
        image_path=cfg.dataset.img_path_test,
        processor=processor,
        ratio=1,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # for r in range(1):
    #     # Randomly generate label_embeddings of size [50, 512]
    #     label_embeddings = torch.randn(87, 32, dtype=torch.float32, device=device)
    #     metrics_total, best_label_tti, best_label_itt = inference_test(
    #         model,
    #         processor,
    #         dataloader=test_dataloader,
    #         label_embeddings=label_embeddings,
    #         epoch=0,
    #         device=device,
    #         inspect_labels=True,
    #     )


@hydra.main(config_path="../../configs", config_name="flickr30k", version_base=None)
def main(cfg):
    test(cfg)


if __name__ == "__main__":
    main()
