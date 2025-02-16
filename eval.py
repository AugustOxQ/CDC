import json
import os
import random
import warnings
from datetime import datetime
from enum import unique
from turtle import update

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from altair import sample
from omegaconf import DictConfig, OmegaConf
from pyparsing import WordStart
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import all_
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer

# Import local packages
from src.data.cdc_datamodule import CDC_test
from src.data.cdc_datamodule import CDC_train_preextract as CDC_train
from src.data.cdc_datamodule import FeatureExtractionDataset
from src.metric.loss import ContrastiveLoss, CosineLoss, MeanSquareLoss
from src.models.cdc import CDC
from src.models.components.clustering import Clustering
from src.utils import (
    EmbeddingManager,
    FeatureManager,
    FolderManager,
    calculate_n_clusters,
    evalrank,
    plot_umap,
)

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_recall_at_k(similarities, k):
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    correct_at_k = np.sum(top_k_indices == np.arange(similarities.shape[0])[:, None])
    return correct_at_k / similarities.shape[0]


def retrieve_top_k(image_embeddings, text_embeddings, top_k=5):
    # Normalize the embeddings to ensure cosine similarity works as expected
    image_embeddings = F.normalize(image_embeddings, p=2, dim=1)  # (1000, 512)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)  # (5000, 512)

    # Compute the cosine similarity matrix between images and texts
    similarity_matrix = torch.mm(image_embeddings, text_embeddings.t())  # (1000, 5000)

    # Image-to-Text Retrieval: For each image, get the top-k most similar text embeddings
    _, topk_text_indices = torch.topk(similarity_matrix, top_k, dim=1)

    # Text-to-Image Retrieval: For each text, get the top-k most similar image embeddings
    _, topk_image_indices = torch.topk(similarity_matrix.t(), top_k, dim=1)

    return topk_text_indices, topk_image_indices


def inference_test(
    model,
    tokenizer,
    dataloader,
    label_embeddings,
    device,
    epoch=0,
    Ks=[1, 5, 10],
    top_k=5,
):
    # Load unique label embeddings up to a reasonable number of labels
    label_embeddings = label_embeddings[:300]

    total_samples = 0
    total_better_count = 0
    total_improvement = 0.0
    total_worst_improvement = 0.0

    all_img_emb = []
    all_txt_emb = []
    all_comb_emb = {}
    all_best_comb_emb = []

    #  (as there are multiple pieces of text for each image)
    image_to_text_map = []

    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []

    text_index = 0
    image_index = 0

    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(dataloader)):
            image, raw_text = batch
            image_input = image.to(device)
            batch_size = image_input["pixel_values"].size(0)
            raw_text_list = []
            batch_size, captions_per_image = image["pixel_values"].shape[0], 5

            # Flatten raw_text
            for b in range(batch_size):
                for i in range(captions_per_image):
                    raw_text_list.append(raw_text[i][b])
            raw_text = raw_text_list

            # Tokenize raw_text
            text_input = tokenizer(
                raw_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            # Update text_to_image_map and image_to_text_map for this batch
            for batch_id in range(batch_size):
                # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                text_indices = list(range(text_index, text_index + captions_per_image))
                image_to_text_map.append(text_indices)
                text_index += captions_per_image

                # Each of the next captions_per_image text captions correspond to the same image
                text_to_image_map += [image_index] * captions_per_image
                image_index += 1

            # text_input = torch.flatten(text_input, start_dim=0, end_dim=1)

            img_emb, txt_emb = model.encode_img_txt(image_input, text_input)

            # TODO: Save per label embeddings and do Oracle test
            # Convert PyTorch tensors to NumPy arrays
            img_emb_np = img_emb.cpu().numpy()
            txt_emb_np = txt_emb.cpu().numpy()

            best_cosine_sim = np.ones((batch_size, captions_per_image)) * -1  # Initialize to -1
            wost_cosine_sim = np.ones((batch_size, captions_per_image)) * 1  # Initialize to 1
            best_comb_emb = torch.zeros((batch_size, captions_per_image, label_embeddings.size(1)))

            for label_indice, label_embedding in enumerate(label_embeddings):
                label_embedding = label_embedding.to(device)
                comb_emb = model.combine(
                    txt_emb, label_embedding.unsqueeze(0).expand(txt_emb.size(0), -1)
                )

                top_k_text_indices, top_k_image_indices = retrieve_top_k(img_emb, comb_emb, 10)

                torch.save(
                    top_k_text_indices,
                    f"other/retrieval_res/top_k_text_indices_{label_indice}.pt",
                )
                torch.save(
                    top_k_image_indices,
                    f"other/retrieval_res/top_k_image_indices_{label_indice}.pt",
                )

                # Calculate cosine similarity within batch using np instead of torch to save memory
                comb_emb_np = comb_emb.cpu().numpy()
                cosine_sim_comb = cosine_similarity(img_emb_np, comb_emb_np)

                # Update best cosine similarity and corresponding label embeddings
                for i in range(batch_size):
                    for j in range(5):
                        current_cosine_sim = cosine_sim_comb[i, i * 5 + j]
                        if current_cosine_sim > best_cosine_sim[i, j]:
                            best_cosine_sim[i, j] = current_cosine_sim
                            best_comb_emb[i, j] = comb_emb.cpu()[i * 5 + j]
                        if current_cosine_sim < wost_cosine_sim[i, j]:
                            wost_cosine_sim[i, j] = current_cosine_sim

                    # max_cosine_sim_comb = np.max(cosine_sim_comb[i * 5 + j])

            # Compare best cosine similarity with raw cosine similarity
            cosine_sim_raw = np.array(
                [
                    cosine_similarity(
                        img_emb_np[i : i + 1], txt_emb_np[i * 5 : (i + 1) * 5]
                    ).flatten()
                    for i in range(batch_size)
                ]
            )

            total_better_count += np.sum(best_cosine_sim > cosine_sim_raw)
            total_samples += batch_size

            # Calculate improvement
            improvement = (best_cosine_sim - cosine_sim_raw) / cosine_sim_raw
            total_improvement += np.sum(improvement)

            # Calculate worst improvement
            worst_improvement = (best_cosine_sim - wost_cosine_sim) / cosine_sim_raw
            total_worst_improvement += np.sum(worst_improvement)

            # Accumulate embeddings for recall calculation
            all_img_emb.append(img_emb.cpu())
            all_txt_emb.append(txt_emb.cpu())
            all_best_comb_emb.append(best_comb_emb)

            del img_emb, txt_emb, image_input, text_input, comb_emb, label_embedding

            torch.cuda.empty_cache()

    # Concatenate all accumulated embeddings
    all_img_emb = torch.cat(all_img_emb, axis=0)
    all_txt_emb = torch.cat(all_txt_emb, axis=0)
    all_best_comb_emb = torch.cat(all_best_comb_emb, axis=0)

    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)

    all_img_emb /= torch.norm(all_img_emb, dim=1, keepdim=True)
    all_txt_emb /= torch.norm(all_txt_emb, dim=1, keepdim=True)
    all_best_comb_emb /= torch.norm(all_best_comb_emb, dim=1, keepdim=True)
    all_best_comb_emb = all_best_comb_emb.view(-1, all_best_comb_emb.size(2))

    # Compute cosine similarities globally
    cosine_sim_raw_global = cosine_similarity(all_img_emb, all_txt_emb)
    cosine_sim_comb_global = cosine_similarity(all_img_emb, all_best_comb_emb)

    # Compute recall@k globally
    recalls_raw = {k: compute_recall_at_k(cosine_sim_raw_global, k) for k in [1, 5, 10]}
    recalls_comb = {k: compute_recall_at_k(cosine_sim_comb_global, k) for k in [1, 5, 10]}

    # Compute and print aggregated results
    better_percentage = total_better_count / total_samples * 100
    avg_improvement = total_improvement / total_samples * 100
    avg_worst_improvement = total_worst_improvement / total_samples * 100
    print(f"Inference Test - Percentage of better cosine similarity: {better_percentage:.2f}%")
    print(
        f"Inference Test - Average improvement over raw cosine similarity: {avg_improvement:.2f}%"
    )
    print(
        f"Inference Test - Average worst improvement over raw cosine similarity: {avg_worst_improvement:.2f}%"
    )

    metrics_raw = evalrank(all_img_emb, all_txt_emb, text_to_image_map, image_to_text_map, "raw")

    metrics_comb = evalrank(
        all_img_emb, all_best_comb_emb, text_to_image_map, image_to_text_map, "comb"
    )

    metrics_basic = {
        "test/better_percentage": better_percentage,
        "test/avg_improvement": avg_improvement,
        "test/avg_worst_improvement": avg_worst_improvement,
    }

    metrics_total = {**metrics_basic, **metrics_raw, **metrics_comb}

    return metrics_total


def run(cfg: DictConfig, **kwargs):
    # Get args
    model_path = kwargs.get("model_path", "")
    embedding_path = kwargs.get("embedding_path", "")
    experiment_dir = kwargs.get("experiment_dir", "")
    unique_embeddings_path = kwargs.get("unique_embeddings_path", "")

    # Load CDC model

    print(f"Loading model from {model_path}")
    model = CDC()
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model = model.to(device)
    preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Load FolderManager
    folder_manager = FolderManager(base_log_dir=cfg.dataset.log_path)
    _, _ = folder_manager.load_experiment(experiment_dir=experiment_dir)

    # Load FeatureManager
    feature_manager = FeatureManager(cfg.dataset.extract_path, chunk_size=cfg.train.batch_size)
    sample_ids_list = torch.load(
        os.path.join(cfg.dataset.extract_path, "sample_ids_list.pt"), weights_only=False
    )
    feature_manager.load_features()

    # Load EmbeddingManager
    annotations = json.load(open(cfg.dataset.train_path))
    annotations = annotations[: int(len(annotations) * cfg.dataset.ratio)]
    embedding_manager = EmbeddingManager(
        annotations,
        embedding_dim=512,
        chunk_size=cfg.train.batch_size,
        hdf5_dir=embedding_path,
        sample_ids_list=sample_ids_list,
        load_existing=True,
    )

    # Load CDC test dataset
    test_dataset = CDC_test(
        annotation_path=cfg.dataset.test_path,
        image_path=cfg.dataset.img_path,
        preprocess=preprocess,
        ratio=1,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # # Load label embeddings
    # print("Loading label embeddings")
    # label_embedding = embedding_manager.get_all_embeddings()[1]
    # print(f"Label embeddings shape: {label_embedding.size(0)}")
    # unique_embeddings, _ = torch.unique(
    #     label_embedding, return_inverse=True, dim=0
    # )
    unique_embeddings = torch.load(unique_embeddings_path, weights_only=False)
    print(f"Unique embeddings: {unique_embeddings.size(0)}")

    print("Starting inference test")
    inf_test_log = inference_test(model, tokenizer, test_dataloader, unique_embeddings, device)


@hydra.main(config_path="configs", config_name="flickr30k", version_base=None)
def main(cfg):
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set path
    experiment_dir = "res/20240904_102707_flickr30k-preextracted"
    model_path = os.path.join(experiment_dir, "final_model.pth")
    embedding_path = os.path.join(experiment_dir, "init")
    unique_embeddings_path = os.path.join(experiment_dir, "unique_embeddings.pt")
    logger = run(
        cfg=cfg,
        model_path=model_path,
        embedding_path=embedding_path,
        experiment_dir=experiment_dir,
        unique_embeddings_path=unique_embeddings_path,
    )


if __name__ == "__main__":
    main()
