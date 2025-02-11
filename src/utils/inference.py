import os
import time
import warnings

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import local packages
from src.data.cdc_datamodule import FeatureExtractionDataset
from src.utils import compute_metric_difference, eval_rank_oracle, evalrank_all

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
def inference_test(model, processor, dataloader, label_embeddings, epoch, device):
    # Load unique label embeddings up to 300
    label_embeddings = label_embeddings[:300]
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

    metrics_oracle = eval_rank_oracle(
        model,
        label_embeddings,  # shape: (N_label, label_dim)
        all_img_emb,  # shape: (N_img, emb_dim)
        all_txt_emb,  # shape: (N_txt, txt_emb_dim)
        all_txt_full,  # shape: (N_txt, other_dim)  额外文本信息
        text_to_image_map,
        image_to_text_map,
        "oracle",
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

    del (
        all_img_emb,
        all_txt_emb,
        all_txt_full,
        text_to_image_map,
        image_to_text_map,
        metrics_oracle,
        metrics_raw,
        metrics_diff,
    )

    return metrics_total


@hydra.main(config_path="configs", config_name="flickr30k", version_base=None)
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

    for r in range(3):
        # Randomly generate label_embeddings of size [50, 512]
        label_embeddings = torch.randn(87, 32, dtype=torch.float32, device=device)
        inference_test(
            model,
            processor,
            dataloader=test_dataloader,
            label_embeddings=label_embeddings,
            epoch=r,
            device=device,
        )


@hydra.main(config_path="../../configs", config_name="flickr30k", version_base=None)
def main(cfg):
    test(cfg)


if __name__ == "__main__":
    main()


# def inference_test_old(model, processor, dataloader, label_embeddings, epoch, device):
#     # Load unique label embeddings up to 50
#     label_embeddings = label_embeddings[:50]
#     all_img_emb = []
#     all_txt_emb = []
#     all_txt_full = []

#     metrics_best = None
#     metrics_worst = None
#     metrics_best2 = None
#     metrics_worst2 = None

#     #  (as there are multiple pieces of text for each image)
#     image_to_text_map = []
#     # text_to_image_map[i] gives the corresponding image index for the ith text
#     text_to_image_map = []

#     text_index = 0
#     image_index = 0

#     # Accumulate embeddings for recall calculation
#     model.eval()
#     with torch.no_grad():
#         for _, batch in enumerate(tqdm(dataloader)):
#             image, raw_text = batch
#             image_input = image.to(device)
#             batch_size = image_input["pixel_values"].size(0)
#             raw_text_list = []
#             batch_size, captions_per_image = (
#                 image["pixel_values"].shape[0],
#                 dataloader.dataset.captions_per_image,
#             )

#             # Flatten raw_text
#             for b in range(batch_size):
#                 if captions_per_image == 1:
#                     raw_text_list.append(raw_text[b])
#                 else:
#                     for i in range(captions_per_image):
#                         raw_text_list.append(raw_text[i][b])
#             raw_text = raw_text_list

#             # Tokenize raw_text
#             text_input = processor(
#                 text=raw_text,
#                 return_tensors="pt",
#                 padding="max_length",
#                 truncation=True,
#                 max_length=77,
#             ).to(device)

#             # Update text_to_image_map and image_to_text_map for this batch
#             for _ in range(batch_size):
#                 # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
#                 text_indices = list(range(text_index, text_index + captions_per_image))
#                 image_to_text_map.append(text_indices)
#                 text_index += captions_per_image

#                 # Each of the next captions_per_image text captions correspond to the same image
#                 text_to_image_map += [image_index] * captions_per_image
#                 image_index += 1

#             img_emb, txt_emb, _, txt_full = model.module.encode_img_txt(
#                 image_input, text_input
#             )

#             all_img_emb.append(img_emb.cpu())
#             all_txt_emb.append(txt_emb.cpu())
#             all_txt_full.append(txt_full.cpu())

#             del img_emb, txt_emb, image_input, text_input, txt_full

#             torch.cuda.empty_cache()

#     # Concate, normalize, and transform embeddings
#     all_img_emb = torch.cat(all_img_emb, axis=0)  # type: ignore
#     all_txt_emb = torch.cat(all_txt_emb, axis=0)  # type: ignore
#     all_txt_full = torch.cat(all_txt_full, axis=0)  # type: ignore

#     text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
#     image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

#     print("Oracle test: text-to-image")
#     start_time = time.time()
#     best_label_indices, worst_label_idx = oracle_test_tti(
#         model,
#         label_embeddings,
#         all_img_emb,
#         all_txt_emb,
#         all_txt_full,
#         text_to_image_map,
#         device,
#     )
#     end_time = time.time()
#     print(f"Oracle test time: {end_time - start_time}")

#     # Get the best label embeddings
#     best_label_embedding = [label_embeddings[bi] for bi in best_label_indices]
#     worst_label_embedding = [label_embeddings[bi] for bi in worst_label_idx]
#     with torch.no_grad():
#         best_label_embedding = torch.stack(best_label_embedding).to(device)
#         print(f"Best label embedding shape: {best_label_embedding.shape}")
#         all_best_comb_emb = model.module.combine(
#             all_txt_emb.to(device), all_txt_full.to(device), best_label_embedding
#         )
#         worst_label_embedding = torch.stack(worst_label_embedding).to(device)
#         print(f"Worst label embedding shape: {worst_label_embedding.shape}")
#         all_worst_comb_emb = model.module.combine(
#             all_txt_emb.to(device), all_txt_full.to(device), worst_label_embedding
#         )

#     # Normalize embeddings
#     all_best_comb_emb = all_best_comb_emb.to(all_img_emb.device)
#     all_best_comb_emb /= torch.norm(all_best_comb_emb, dim=1, keepdim=True)
#     all_worst_comb_emb = all_worst_comb_emb.to(all_img_emb.device)
#     all_worst_comb_emb /= torch.norm(all_worst_comb_emb, dim=1, keepdim=True)

#     # Evaluate the embeddings
#     metrics_best = evalrank_all(
#         all_img_emb, all_best_comb_emb, text_to_image_map, image_to_text_map, "best"
#     )
#     metrics_worst = evalrank_all(
#         all_img_emb, all_worst_comb_emb, text_to_image_map, image_to_text_map, "worst"
#     )

#     # print("Oracle test: image-to-text")
#     # start_time = time.time()
#     # best_label_indices, worst_label_idx = oracle_test_itt(
#     #     model,
#     #     label_embeddings,
#     #     all_img_emb,
#     #     all_txt_emb,
#     #     all_txt_full,
#     #     image_to_text_map,
#     #     device,
#     # )
#     # end_time = time.time()
#     # print(f"Oracle test time: {end_time - start_time}")

#     # # Get the best label embeddings
#     # best_label_embedding = [label_embeddings[bi] for bi in best_label_indices]
#     # worst_label_embedding = [label_embeddings[bi] for bi in worst_label_idx]
#     # with torch.no_grad():
#     #     best_label_embedding = torch.stack(best_label_embedding).to(device)
#     #     print(f"Best label embedding shape: {best_label_embedding.shape}")
#     #     all_best_comb_emb2 = model.module.combine(
#     #         all_txt_emb.to(device), all_txt_full.to(device), best_label_embedding
#     #     )
#     #     worst_label_embedding = torch.stack(worst_label_embedding).to(device)
#     #     print(f"Worst label embedding shape: {worst_label_embedding.shape}")
#     #     all_worst_comb_emb2 = model.module.combine(
#     #         all_txt_emb.to(device), all_txt_full.to(device), worst_label_embedding
#     #     )

#     # # # Normalize embeddings
#     # all_img_emb /= torch.norm(all_img_emb, dim=1, keepdim=True)
#     # all_txt_emb /= torch.norm(all_txt_emb, dim=1, keepdim=True)
#     # all_best_comb_emb2 = all_best_comb_emb2.to(all_img_emb.device)
#     # all_best_comb_emb2 /= torch.norm(all_best_comb_emb2, dim=1, keepdim=True)
#     # all_worst_comb_emb2 = all_worst_comb_emb2.to(all_img_emb.device)
#     # all_worst_comb_emb2 /= torch.norm(all_worst_comb_emb2, dim=1, keepdim=True)

#     # # Evaluate the embeddings
#     # metrics_best2 = evalrank_all(
#     #     all_img_emb, all_best_comb_emb2, text_to_image_map, image_to_text_map, "best2"
#     # )
#     # metrics_worst2 = evalrank_all(
#     #     all_img_emb, all_worst_comb_emb2, text_to_image_map, image_to_text_map, "worst2"
#     # )

#     metrics_raw = evalrank_all(
#         all_img_emb, all_txt_emb, text_to_image_map, image_to_text_map, "raw"
#     )

#     # Difference between two dictionaries
#     metrics_diff = compute_metric_difference(metrics_best, metrics_raw, "raw", "diff")
#     # metrics_diff2 = compute_metric_difference(
#     #     metrics_best2, metrics_raw, "raw", "diff2"
#     # )

#     metrics_total = {
#         "test/epoch": epoch,
#         **metrics_raw,
#         **metrics_best,
#         **metrics_worst,
#         # **metrics_best2,
#         # **metrics_worst2,
#         **metrics_diff,
#         # **metrics_diff2,
#     }

#     return metrics_total

# def oracle_test_tti(
#     model, label_embeddings, img_emb, txt_emb, txt_full, text_to_image_map, device
# ):
#     """This uses the oracle method to find the best label embedding for each text by computing the
#     text-image recall@k."""
#     # Load unique label embeddings up to 50
#     label_embeddings = label_embeddings[:50].to(device)

#     num_images = img_emb.size(0)  # 1000 images
#     num_texts = txt_full.size(0)  # 5000 texts

#     # To store the best label embedding index for each text
#     best_label_indices = torch.zeros(num_texts, dtype=torch.int32)
#     worst_label_indices = torch.ones(num_texts, dtype=torch.int32)

#     # To store the recall sum (r-sum) for evaluation
#     total_r_sum = 0.0
#     total_r_sum_worst = 0.0

#     img_emb = img_emb.to(device)
#     txt_emb = txt_emb.to(device)
#     txt_full = txt_full.to(device)

#     with torch.no_grad():
#         # Iterate over each text
#         for text_id in tqdm(range(num_texts)):
#             # Get the correct image index for this text
#             correct_image_idx = text_to_image_map[text_id].item()

#             # Variable to track the best recall and label embedding index for this text
#             best_rank = num_texts  # Initialize to a high value
#             worst_rank = 1  # Initialize to a low value
#             best_label_idx = 0  # Initialize best label embedding index
#             worst_label_idx = 0  # Initialize worst label embedding index

#             # Iterate over each label embedding
#             for label_idx, label_embedding in enumerate(label_embeddings):
#                 # Expand the label embedding for the current text embedding
#                 expanded_label_emb = label_embedding.unsqueeze(0).expand(
#                     txt_full[text_id : text_id + 1].size(0), -1
#                 )

#                 # Combine text embedding with label embedding
#                 comb_emb = model.module.combine(
#                     txt_emb[text_id : text_id + 1],
#                     txt_full[text_id : text_id + 1],
#                     expanded_label_emb,
#                 ).to(device)

#                 # Normalize combined embedding
#                 comb_emb /= torch.norm(comb_emb, dim=1, keepdim=True)

#                 # Compute cosine similarity between the combined text embedding and all image embeddings
#                 cosine_sim_comb = torch.mm(comb_emb, img_emb.T).flatten()

#                 # Get the recall r-sum: find the ranking of the correct image for the current text
#                 sorted_sim_indices = torch.argsort(cosine_sim_comb, descending=True)
#                 rank_of_correct_image = (
#                     (sorted_sim_indices == correct_image_idx)
#                     .nonzero(as_tuple=True)[0]
#                     .item()
#                 )

#                 # Recall is better if the correct image is ranked lower (closer to 1)
#                 rank = rank_of_correct_image + 1  # Higher r_sum is better

#                 # If this label embedding gives a better recall (lower rank), update
#                 if rank <= best_rank:
#                     best_rank = rank
#                     best_label_idx = label_idx

#                 # If this label embedding gives a worse recall (higher rank), update
#                 if rank >= worst_rank:
#                     worst_rank = rank
#                     worst_label_idx = label_idx

#                 # Clean up intermediate tensors explicitly
#                 comb_emb.cpu()  # Move to CPU
#                 cosine_sim_comb.cpu()  # Move to CPU
#                 sorted_sim_indices.cpu()  # Move to CPU

#             # After iterating over all label embeddings, store the best label index for this text
#             best_label_indices[text_id] = best_label_idx
#             total_r_sum += best_rank

#             # Store the worst label index for this text
#             worst_label_indices[text_id] = worst_label_idx
#             total_r_sum_worst += worst_rank

#             # Clean memory
#             del best_rank, best_label_idx, worst_label_idx
#             torch.cuda.empty_cache()

#     # Return the best label indices per text and the total r-sum (sum of all recall scores)
#     print(f"Total rank: {total_r_sum}")
#     print(f"Total rank (worst): {total_r_sum_worst}")

#     # Count how many times the best label index is different from the worst label index
#     different_indices = torch.sum(
#         best_label_indices[:500] != worst_label_indices[:500]
#     ).item()
#     print(f"Different indices: {different_indices}")

#     return best_label_indices, worst_label_indices


# def oracle_test_itt(
#     model, label_embeddings, img_emb, txt_emb, txt_full, image_to_text_map, device
# ):
#     """This uses the oracle method for image-to-text retrieval by precomputing all possible
#     combinations of text embeddings and label embeddings."""
#     # Load unique label embeddings up to 50
#     label_embeddings = label_embeddings[:50].to(device)

#     num_images = img_emb.size(0)  # 5000 images
#     num_texts = txt_full.size(0)  # 5000 texts
#     num_labels = label_embeddings.size(0)  # Number of label embeddings
#     img_emb, txt_full = img_emb.to(device), txt_full.to(device)
#     txt_emb = txt_emb.to(device)

#     # Precompute all combinations of text and label embeddings
#     combined_txt_label_emb = torch.zeros(
#         (num_texts, num_labels, txt_full.size(-1)), device=device
#     )

#     print("Precomputing text-label embedding combinations...")
#     with torch.no_grad():
#         for label_idx, label_embedding in enumerate(tqdm(label_embeddings)):
#             expanded_label_emb = label_embedding.unsqueeze(0).expand(num_texts, -1)
#             combined_txt_label_emb[:, label_idx, :] = model.module.combine(
#                 txt_emb, txt_full, expanded_label_emb
#             )

#     # Normalize combined embeddings
#     combined_txt_label_emb /= torch.norm(combined_txt_label_emb, dim=2, keepdim=True)

#     # To store the best and worst label embedding index for each image
#     best_label_indices = torch.zeros(num_images, dtype=torch.int32)
#     worst_label_indices = torch.ones(num_images, dtype=torch.int32)

#     # To store the recall sum (r-sum) for evaluation
#     total_r_sum = 0.0
#     total_r_sum_worst = 0.0

#     img_emb = img_emb.to(device)

#     with torch.no_grad():
#         # Iterate over each image
#         for img_id in tqdm(range(num_images)):
#             # Get the correct text indices for this image
#             correct_text_indices = image_to_text_map[
#                 img_id
#             ].tolist()  # Text indices associated with the image

#             # Variable to track the best and worst recall for this image
#             best_rank = num_texts  # Initialize to a high value
#             worst_rank = 1  # Initialize to a low value
#             best_label_idx = 0  # Initialize best label embedding index
#             worst_label_idx = 0  # Initialize worst label embedding index

#             # Stack all text-label combinations for this image
#             combined_sims = torch.zeros(num_texts * num_labels, device=device)

#             # Compute cosine similarity between the image and all text-label combinations
#             for label_idx in range(num_labels):
#                 combined_sims[label_idx * num_texts : (label_idx + 1) * num_texts] = (
#                     torch.mm(
#                         img_emb[img_id : img_id + 1],
#                         combined_txt_label_emb[:, label_idx, :].T,
#                     ).flatten()
#                 )

#             # Rank the correct text embeddings for each label embedding
#             for label_idx in range(num_labels):
#                 sim_slice = combined_sims[
#                     label_idx * num_texts : (label_idx + 1) * num_texts
#                 ]
#                 sorted_sim_indices = torch.argsort(sim_slice, descending=True)

#                 # Find the minimum rank of the correct texts for the current label
#                 min_rank_of_correct_text = min(
#                     [
#                         (sorted_sim_indices == idx).nonzero(as_tuple=True)[0].item()
#                         for idx in correct_text_indices
#                     ]
#                 )

#                 rank = min_rank_of_correct_text + 1  # Rank starts from 1

#                 # If this label embedding gives a better recall (lower rank), update
#                 if rank <= best_rank:
#                     best_rank = rank
#                     best_label_idx = label_idx

#                 # If this label embedding gives a worse recall (higher rank), update
#                 if rank >= worst_rank:
#                     worst_rank = rank
#                     worst_label_idx = label_idx

#             # After iterating over all label embeddings, store the best and worst label indices for this image
#             best_label_indices[img_id] = best_label_idx
#             total_r_sum += best_rank

#             worst_label_indices[img_id] = worst_label_idx
#             total_r_sum_worst += worst_rank

#             # Clean memory
#             del combined_sims
#             torch.cuda.empty_cache()

#     # Return the best label indices per image and the total r-sum (sum of all recall scores)
#     print(f"Total r-sum: {total_r_sum}")
#     print(f"Total r-sum (worst): {total_r_sum_worst}")

#     # Count how many times the best label index is different from the worst label index
#     different_indices = torch.sum(best_label_indices != worst_label_indices).item()
#     print(f"Different indices: {different_indices}")

#     return best_label_indices, worst_label_indices


# # Calculate cosine similarity within batch
#             # Calculate cosine similarity between image and text embeddings
#             cosine_sim_raw = cosine_similarity(
#                 img_emb.cpu().numpy(), txt_emb_cls.cpu().numpy()
#             ).diagonal()

#             # Calculate cosine similarity between image and combined embeddings
#             cosine_sim_comb = cosine_similarity(
#                 img_emb.cpu().numpy(), comb_emb.cpu().numpy()
#             ).diagonal()

#             # Calculate cosine similarity between image and combined embeddings (shuffled)
#             cosine_sim_comb_shuffled = cosine_similarity(
#                 img_emb.cpu().numpy(), comb_emb_shuffled.cpu().numpy()
#             ).diagonal()

#             # Test 1: Whether cosine_sim_comb is equal or greater than cosine_sim_raw
#             comparison_raw = cosine_sim_comb >= cosine_sim_raw
#             raw_better_count = np.sum(comparison_raw)
#             total_raw_better_count += raw_better_count

#             # Test 2: Whether cosine_sim_comb is equal or greater than cosine_sim_comb_shuffled
#             comparison_shuffled = cosine_sim_comb >= cosine_sim_comb_shuffled
#             shuffled_better_count = np.sum(comparison_shuffled)
#             total_shuffled_better_count += shuffled_better_count

#             # Test 3: Diversity of label embeddings
#             unique_embeddings = torch.unique(label_embedding, dim=0)
#             normalized_embeddings = F.normalize(unique_embeddings, p=2, dim=1)
#             similarity_matrix = torch.mm(
#                 normalized_embeddings, normalized_embeddings.t()
#             )
#             mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
#             pairwise_similarities = similarity_matrix[mask]
#             total_diversity += 1 - torch.mean(pairwise_similarities).item()

#             # Sample size
#             batch_size = cosine_sim_comb.shape[0]
#             total_samples += batch_size
