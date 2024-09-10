import json
import os
from turtle import up, update
import warnings
import time
from datetime import datetime
from altair import sample
import numpy as np
from pyparsing import WordStart
from sklearn import metrics
from sklearn.cluster import k_means
from sklearn.metrics.pairwise import cosine_similarity
import random

import hydra
from sqlalchemy import all_
import wandb
import omegaconf

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor

# Import local packages
from src.data.cdc_datamodule import CDC_test, FeatureExtractionDataset
from src.data.cdc_datamodule import CDC_train_preextract as CDC_train
from src.metric.loss import CosineLoss, MeanSquareLoss, ContrastiveLoss
from src.models.cdc import CDC
from src.models.components.clustering import Clustering
from src.utils import EmbeddingManager, FolderManager, evalrank, calculate_n_clusters, plot_umap, FeatureManager

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
def k_means_stop_condition():
    #TODO: Keep track of the diversity of the k-means centroids by calculating the mean of pairwise distances between the centroids. This can be an automatic way of deciding when to stop the clustering process.
    
    pass


def random_sample_with_replacement(label_embedding):
    size = label_embedding.size(0)
    random_indices = torch.randint(0, size, (size,))
    
    # Ensure that sampled index is not the same as the original
    for i in range(size):
        while random_indices[i] == i:
            random_indices[i] = torch.randint(0, size, (1,))
    
    return label_embedding[random_indices]


def compute_recall_at_k(similarities, k):
    top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
    correct_at_k = np.sum(top_k_indices == np.arange(similarities.shape[0])[:, None])
    return correct_at_k / similarities.shape[0]


def extract_and_store_features(annotation_path, image_path, feature_manager, batch_size, model, preprocess, tokenizer, device, ratio=0.1):

    dataset = FeatureExtractionDataset(annotation_path, image_path, preprocess, ratio=ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    sample_ids_list = []
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(dataloader)):
            images, raw_texts, sample_ids = batch
            
            image_input = images.to(device)
            text_input = tokenizer(
                raw_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            with torch.no_grad():
                img_emb, txt_emb = model.encode_img_txt(image_input, text_input)
                img_emb, txt_emb = img_emb.cpu().numpy(), txt_emb.cpu().numpy()

            feature_manager.add_features_chunk(batch_id, img_emb, txt_emb, sample_ids)
            
            sample_ids_list.extend(sample_ids)
            
        return sample_ids_list
                
    # feature_manager.debug_print()
    
def inference_train(model, tokenizer, dataloader, device, epoch=0, Ks=[1, 5, 10], max_batches=10):
    # Read embeddings directly from the dataloader, compare with other mebeddings from the same batch
    model.eval()
    total_raw_better_count = 0
    total_shuffled_better_count = 0
    total_diversity = 0.0
    total_samples = 0
    # total_precisions = {k: 0.0 for k in Ks}

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(dataloader)):
            
            # Limit the number of batches to process
            if batch_id >= max_batches:
                print(f"Epoch {epoch}: Stopping inference after {max_batches} batches")
                break
            
            img_emb, txt_emb, label_embedding, sample_id = batch
            img_emb, txt_emb, label_embedding = img_emb.squeeze(0), txt_emb.squeeze(0), label_embedding.squeeze(0)
            
            img_emb = img_emb.to(device)
            txt_emb = txt_emb.to(device)
            label_embedding = label_embedding.to(device)
            
            # Shuffle label embeddings
            label_embedding_shuffled = random_sample_with_replacement(label_embedding)
            
            # Combine embeddings
            comb_emb = model.combine(txt_emb, label_embedding)
            
            # Combine embeddings (shuffled)
            comb_emb_shuffled = model.combine(
                txt_emb, label_embedding_shuffled
            )
            
            # Calculate cosine similarity within batch
            # Calculate cosine similarity between image and text embeddings
            cosine_sim_raw = cosine_similarity(
                img_emb.cpu().numpy(),
                txt_emb.cpu().numpy()
            ).diagonal()
            
            # Calculate cosine similarity between image and combined embeddings
            cosine_sim_comb = cosine_similarity(
                img_emb.cpu().numpy(),
                comb_emb.cpu().numpy()
            ).diagonal()
            
            # Calculate cosine similarity between image and combined embeddings (shuffled)
            cosine_sim_comb_shuffled = cosine_similarity(
                img_emb.cpu().numpy(),
                comb_emb_shuffled.cpu().numpy()
            ).diagonal()
            
            # Test 1: Whether cosine_sim_comb is greater than cosine_sim_raw
            comparison_raw = cosine_sim_comb > cosine_sim_raw
            raw_better_count = np.sum(comparison_raw)
            total_raw_better_count += raw_better_count

            # Test 2: Whether cosine_sim_comb is greater than cosine_sim_comb_shuffled
            comparison_shuffled = cosine_sim_comb > cosine_sim_comb_shuffled
            shuffled_better_count = np.sum(comparison_shuffled)
            total_shuffled_better_count += shuffled_better_count
            
            # Test 3: Diversity of label embeddings
            normalized_embeddings = F.normalize(label_embedding, p=2, dim=1)
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
            mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
            pairwise_similarities = similarity_matrix[mask]
            total_diversity += 1 - torch.mean(pairwise_similarities).item()
            
            # Sample size
            batch_size = cosine_sim_comb.shape[0]
            total_samples += batch_size
                        
            del img_emb, txt_emb, label_embedding, comb_emb, comb_emb_shuffled, label_embedding_shuffled
            
            torch.cuda.empty_cache()

    
    # Calculate percentage of better label embeddings
    raw_better_percentage = total_raw_better_count / total_samples * 100
    shuffled_better_percentage = total_shuffled_better_count / total_samples * 100
    diversity_score = total_diversity / total_samples * 100
    
    print(f"Epoch {epoch}: Combined embeddings better than raw embeddings: {raw_better_percentage:.2f}%")
    print(f"Epoch {epoch}: Combined embeddings better than shuffled embeddings: {shuffled_better_percentage:.2f}%")
    print(f"Epoch {epoch}: Diversity score: {diversity_score:.2f}")

    return {
        "val/raw_better_percentage": raw_better_percentage,
        "val/shuffled_better_percentage": shuffled_better_percentage,
        "val/diversity_score": diversity_score,
    }


def oracle_test_tti(model, label_embeddings, img_emb, txt_emb, text_to_image_map, device, Ks=[1, 5, 10]):
    """
    This uses the oracle method to find the best label embedding for each text by computing the text-image recall@k
    """
    # Load unique label embeddings up to 50
    label_embeddings = label_embeddings[:50].to(device)

    num_images = img_emb.size(0)  # 1000 images
    num_texts = txt_emb.size(0)   # 5000 texts
    
    # To store the best label embedding index for each text
    best_label_indices = torch.zeros(num_texts, dtype=torch.int32)
    worst_label_indices = torch.ones(num_texts, dtype=torch.int32)
    
    # To store the recall sum (r-sum) for evaluation
    total_r_sum = 0.0
    total_r_sum_worst = 0.0
    
    img_emb = img_emb.to(device)
    txt_emb = txt_emb.to(device)
    
    with torch.no_grad():
        # Iterate over each text
        for text_id in tqdm(range(num_texts)):
            # Get the correct image index for this text
            correct_image_idx = text_to_image_map[text_id].item()
            
            # Variable to track the best recall and label embedding index for this text
            best_rank = num_texts # Initialize to a high value
            worst_rank = 1  # Initialize to a low value
            best_label_idx = 0  # Initialize best label embedding index
            worst_label_idx = 0  # Initialize worst label embedding index
            
            # Iterate over each label embedding
            for label_idx, label_embedding in enumerate(label_embeddings):
                # Expand the label embedding for the current text embedding
                expanded_label_emb = label_embedding.unsqueeze(0).expand(txt_emb[text_id:text_id+1].size(0), -1)
                
                # Combine text embedding with label embedding
                comb_emb = model.combine(txt_emb[text_id:text_id+1], expanded_label_emb).to(device)
                
                # Normalize combined embedding
                comb_emb /= torch.norm(comb_emb, dim=1, keepdim=True)
                
                # Compute cosine similarity between the combined text embedding and all image embeddings
                cosine_sim_comb = torch.mm(comb_emb, img_emb.T).flatten()
                
                # Get the recall r-sum: find the ranking of the correct image for the current text
                sorted_sim_indices = torch.argsort(cosine_sim_comb, descending=True)
                rank_of_correct_image = (sorted_sim_indices == correct_image_idx).nonzero(as_tuple=True)[0].item()
                
                # Recall is better if the correct image is ranked lower (closer to 1)
                rank = rank_of_correct_image + 1  # Higher r_sum is better
                
                # If this label embedding gives a better recall (lower rank), update
                if rank <= best_rank:
                    best_rank = rank
                    best_label_idx = label_idx
                
                # If this label embedding gives a worse recall (higher rank), update
                if rank >= worst_rank:
                    worst_rank = rank
                    worst_label_idx = label_idx
                    
                # Clean up intermediate tensors explicitly
                comb_emb.cpu()  # Move to CPU
                cosine_sim_comb.cpu()  # Move to CPU
                sorted_sim_indices.cpu()  # Move to CPU
            
            # After iterating over all label embeddings, store the best label index for this text
            best_label_indices[text_id] = best_label_idx
            total_r_sum += best_rank
            
            # Store the worst label index for this text
            worst_label_indices[text_id] = worst_label_idx
            total_r_sum_worst += worst_rank
            
            # Clean memory
            del best_rank, best_label_idx, worst_label_idx
            torch.cuda.empty_cache()
            
            # # Only process the first 500 texts            
            # if text_id == 500:
            #     break
    
    # Return the best label indices per text and the total r-sum (sum of all recall scores)
    print(f"Total r-sum: {total_r_sum}")
    print(f"Total r-sum (worst): {total_r_sum_worst}")
    
    # Count how many times the best label index is different from the worst label index
    different_indices = torch.sum(best_label_indices[:500] != worst_label_indices[:500]).item()
    print(f"Different indices: {different_indices}")
    
    return best_label_indices, worst_label_indices
    

def inference_test(model, tokenizer, dataloader, label_embeddings, device, epoch, Ks=[1, 5, 10]):
    # Load unique label embeddings up to 50
    label_embeddings = label_embeddings[:50]
    all_img_emb = []
    all_txt_emb = []
    
    #  (as there are multiple pieces of text for each image)
    image_to_text_map = []
    # text_to_image_map[i] gives the corresponding image index for the ith text
    text_to_image_map = []
    text_index = 0
    image_index = 0
    
    # Accumulate embeddings for recall calculation
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

            img_emb, txt_emb = model.encode_img_txt(image_input, text_input)
            
            all_img_emb.append(img_emb.cpu())
            all_txt_emb.append(txt_emb.cpu())
            
            del img_emb, txt_emb, image_input, text_input
            
            torch.cuda.empty_cache()
            
    # Concate, normalize, and transform embeddings
    all_img_emb = torch.cat(all_img_emb, axis=0)
    all_txt_emb = torch.cat(all_txt_emb, axis=0)
    text_to_image_map = torch.LongTensor(text_to_image_map).to(device)
    image_to_text_map = torch.LongTensor(image_to_text_map).to(device)
    
    start_time = time.time()
    best_label_indices, worst_label_idx = oracle_test_tti(model, label_embeddings, all_img_emb, all_txt_emb, text_to_image_map, device)
    end_time = time.time()
    print(f"Oracle test time: {end_time - start_time}")
    
    # Get the best label embeddings
    best_label_embedding = [label_embeddings[bi] for bi in best_label_indices]
    worst_label_embedding = [label_embeddings[bi] for bi in worst_label_idx]
    with torch.no_grad():
        best_label_embedding = torch.stack(best_label_embedding).to(device)
        all_best_comb_emb = model.combine(all_txt_emb.to(device), best_label_embedding)
        worst_label_embedding = torch.stack(worst_label_embedding).to(device)
        all_worst_comb_emb = model.combine(all_txt_emb.to(device), worst_label_embedding)
    
    # Normalize embeddings
    all_img_emb /= torch.norm(all_img_emb, dim=1, keepdim=True)
    all_txt_emb /= torch.norm(all_txt_emb, dim=1, keepdim=True)    
    all_best_comb_emb = all_best_comb_emb.to(all_img_emb.device)
    all_best_comb_emb /= torch.norm(all_best_comb_emb, dim=1, keepdim=True)
    all_worst_comb_emb = all_worst_comb_emb.to(all_img_emb.device)
    all_worst_comb_emb /= torch.norm(all_worst_comb_emb, dim=1, keepdim=True)
    
    # Evaluate the embeddings
    metrics_raw = evalrank(all_img_emb, all_txt_emb, text_to_image_map, image_to_text_map, "raw")
    metrics_best = evalrank(all_img_emb, all_best_comb_emb, text_to_image_map, image_to_text_map, "comb")
    metrics_worst = evalrank(all_img_emb, all_worst_comb_emb, text_to_image_map, image_to_text_map, "worst")
    
    metrics_total = {**metrics_raw, **metrics_best, **metrics_worst}
    
    return metrics_total


def train(cfg: DictConfig, **kwargs):
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    train_dataloader = kwargs["train_dataloader"]
    criteria = kwargs["criteria"]
    optimizer = kwargs["optimizer"]
    epoch = kwargs["epoch"]
    scheduler = kwargs["scheduler"]
    embedding_manager = kwargs["embedding_manager"]
    update_label_embedding = kwargs["update_label_embedding"]
    log_interval = cfg.train.log_interval
    wandb_run = kwargs["wandb_run"]

    model.train()
    epoch_metrics = {
        "loss": 0.0,
        "other_metrics": {}
    }
    
    for batch_id, batch in enumerate(tqdm(train_dataloader)):
        
        img_emb, txt_emb, label_embedding, sample_id = batch
        img_emb, txt_emb, label_embedding = img_emb.squeeze(0), txt_emb.squeeze(0), label_embedding.squeeze(0)

        img_emb, txt_emb = img_emb.to(device), txt_emb.to(device)
    
        label_embedding = label_embedding.to(device).clone().detach().requires_grad_(True)
        label_embedding_cp = label_embedding.clone().detach()
        
        # Initialize optimizer for label_embedding
        current_lr = optimizer.param_groups[0]['lr']
        optimizer_label = torch.optim.AdamW([label_embedding], lr=current_lr, weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.betas[0], cfg.train.betas[1]))

        lbl_emb = model.label_encoder(label_embedding)
        comb_emb = model.combine(txt_emb, label_embedding)
        
        loss = criteria(img_emb, comb_emb)
        epoch_metrics["loss"] += loss.item()
        optimizer.zero_grad()
        if update_label_embedding:
            optimizer_label.zero_grad()
        loss.backward()
        optimizer.step()
        if update_label_embedding:
            optimizer_label.step()
        
        # Check if label_embedding is updated
        diff = torch.sum(label_embedding - label_embedding_cp)
        if update_label_embedding:
            assert diff != 0, "Label embedding should be updated after backward pass"
        else:
            assert diff == 0, "Label embedding should not be updated after backward pass"
        
        if update_label_embedding:
            embedding_manager.update_chunk_embeddings(batch_id, sample_id, label_embedding)

        # Log
        scheduler.step(epoch + batch_id / len(train_dataloader))
        if batch_id % log_interval == 0 or batch_id == len(train_dataloader) - 1:
            print(
                f"Epoch: {epoch}, Batch: {batch_id} / {len(train_dataloader)-1 }, Loss: {loss.item()}"
            )
            
        # Wandb logger
        wandb_run.log(
            {   
                "train/total_loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            },
        )

        del (
            img_emb,
            txt_emb,
            lbl_emb,
            comb_emb,
            loss,
            label_embedding,
        )

        torch.cuda.empty_cache()

    return epoch_metrics


def run(cfg: DictConfig, **kwargs):
    # Get args
    logger_dir = kwargs["logger_dir"]

    wandb_run = kwargs["wandb_run"]
    model = CDC().to(device)
    preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize FolderManager
    folder_manager = FolderManager(base_log_dir=cfg.dataset.log_path)
    
    # Initialize feature manager
    feature_manager = FeatureManager(cfg.dataset.extract_path, chunk_size=cfg.train.batch_size)
    
    # Initialize experiment
    experiment_dir, init_dir, plot_dir = folder_manager.initialize_experiment(cfg.log_tag)
    checkpoint_dir, logs_dir = folder_manager.create_directories(experiment_dir)
    
    if cfg.dataset.pre_extract:
        print("##########Extracting and storing features##########")
        sample_ids_list = extract_and_store_features(cfg.dataset.train_path, cfg.dataset.img_path, feature_manager, cfg.train.batch_size, model, preprocess, tokenizer, device, ratio = cfg.dataset.ratio)
        torch.save(sample_ids_list, os.path.join(cfg.dataset.extract_path, "sample_ids_list.pt"))
    else:
        print("##########Loading pre-extracted features##########")
        sample_ids_list = torch.load(os.path.join(cfg.dataset.extract_path, "sample_ids_list.pt"))
        # turn sample_ids_list into a list of integers
        feature_manager.load_features()
        
    sample_ids_list = [int(sample_id) for sample_id in sample_ids_list]

    # Initialize embedding manager
    print("##########Initializing Embedding Manager##########")
    annotations = json.load(open(cfg.dataset.train_path))
    annotations = annotations[: int(len(annotations) * cfg.dataset.ratio)]
    embedding_manager = EmbeddingManager(
        annotations, embedding_dim=512, chunk_size=cfg.train.batch_size, embeddings_dir=init_dir, sample_ids_list=sample_ids_list
    )
    embedding_manager.load_embeddings()

    # Samples to track
    samples_to_track = [0, 1, 2, 3, 4]  # Indices of the samples to track

    # Initialize clustering
    clustering = Clustering()

    # Create Train and Test dataloader
    train_dataset = CDC_train(
        annotation_path=cfg.dataset.train_path,
        image_path=cfg.dataset.img_path,
        embedding_manager=embedding_manager,
        feature_manager=feature_manager,
        ratio=cfg.dataset.ratio,
    )
    
    # batch_size = 1 and no shuffle, just load chunk embeddings
    train_dataloader = DataLoader(
        train_dataset,
        # batch_size=cfg.train.batch_size,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    
    test_dataset = CDC_test(
        annotation_path=cfg.dataset.test_path,
        image_path=cfg.dataset.img_path,
        preprocess=preprocess,
        ratio=1
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # Setup criteria and optimizer and scheduler
    criteria = CosineLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.betas[0], cfg.train.betas[1]),
    )
    # scheduler = CosineAnnealingLR(
    #     optimizer, T_max=cfg.train.T_max, eta_min=cfg.train.lr_min
    # )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.train.warm_up, T_mult=1, eta_min=cfg.train.lr_min
    )

    # Callbacks
    logger = []
    max_epoch = cfg.train.max_epochs
    
    initial_n_clusters = train_dataset.get_len() - cfg.train_2.initial_n_clusters # Initial number of clusters
    first_stage_n = cfg.train_2.first_stage_n # Number of clusters after first stage
    second_stage_n = cfg.train_2.second_stage_n # Number of clusters after second stage
    k_means_start_epoch = cfg.train_2.k_means_start_epoch # Start k-means clustering
    k_means_middle_epoch = cfg.train_2.k_means_middle_epoch # Start slow k-means clustering
    k_means_end_epoch = cfg.train_2.k_means_end_epoch # End k-means clustering
    update_label_embedding = True # Update label embeddings during training
    
    assert k_means_start_epoch <= k_means_end_epoch <= max_epoch, "Invalid epoch values for k-means clustering"
    
    assert initial_n_clusters >= first_stage_n >= second_stage_n, "Invalid number of clusters"

    # Start training
    for epoch in range(max_epoch):
        wandb_run.log({"train/epoch": epoch})
        logger_epoch = {}
        logger_epoch["epoch"] = epoch
        unique_embeddings = None
        
        # Train
        if cfg.control.train: # Network training
            print(f"##########Epoch {epoch}: Training##########")
            
            # # Create new directory for the current epoch
            if cfg.control.save_per_epoch == True:
                new_embeddings_dir = folder_manager.create_epoch_folder(epoch)
                embedding_manager.embeddings_dir = new_embeddings_dir
                embedding_manager.save_embeddings_to_new_folder(new_embeddings_dir)
            embedding_manager.load_embeddings()
            
            if epoch == k_means_end_epoch:
                update_label_embedding = False
                print("##########Cease label embedding updates##########")
            
            train_epoch_log = train(
                cfg,
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                epoch=epoch,
                criteria=criteria,
                optimizer=optimizer,
                embedding_manager=embedding_manager,
                update_label_embedding=update_label_embedding,
                scheduler=scheduler,
                wandb_run=wandb_run,
            )
            scheduler.step()
            logger_epoch["train"] = train_epoch_log
            
            # Save epoch metrics
            folder_manager.save_metrics(train_epoch_log, logs_dir, epoch)
            
        if cfg.control.val:
            print("##########Testing train dataset##########")
            inf_train_log = inference_train(model, tokenizer, train_dataloader, device, epoch, [1, 5, 10])
            wandb_run.log(inf_train_log)
            logger_epoch["inference_train"] = inf_train_log
    
        if cfg.control.train_2: # KMeans update
            n_clusters = calculate_n_clusters(initial_n_clusters, first_stage_n, second_stage_n, epoch, k_means_start_epoch, k_means_end_epoch)
            wandb_run.log({"train/n_clusters": n_clusters})
            
            # Perform clustering and update embeddings by merging
            if k_means_start_epoch <= epoch < k_means_end_epoch:
                print(f"##########Epoch {epoch}: Expected number of clusters: {n_clusters}##########")

                # Load embeddings
                embedding_manager.load_embeddings()
                sample_ids, label_embedding = embedding_manager.get_all_embeddings()

                # Perform UMAP and clustering on unique embeddings
                print("##########Performing UMAP##########")
                umap_features = clustering.get_umap(label_embedding)
                
                print("##########Performing KMeans##########")
                umap_labels, _ = clustering.get_kmeans(
                umap_features, n_clusters=n_clusters
                )

                if cfg.control.save:
                    # Plot UMAP before clustering update
                    umap_features_np = umap_features.cpu().numpy()
                    umap_labels_np = umap_labels.cpu().numpy()
                    plot_umap(
                        umap_features_np, umap_labels_np, plot_dir, epoch, samples_to_track
                    )
                
                print("##########Performing clustering update##########")
                # Map clustering results back to the original embeddings
                updated_embeddings = clustering.kmeans_update(
                    umap_labels=umap_labels,
                    original_embeddings=label_embedding,
                    update_type='hard',
                    alpha=0.1
                )
                
                # Find unique embeddings
                unique_embeddings, _ = torch.unique(
                    updated_embeddings, return_inverse=True, dim=0
                )
                print(f"Unique embeddings after clustering update: {unique_embeddings.size(0)}")
                
                # Save unique embeddings
                torch.save(unique_embeddings[:50], os.path.join(experiment_dir, f"unique_embeddings.pt"))
                
                # Check how many embeddings have been updated by k-means
                differences = torch.any(label_embedding != updated_embeddings, dim=1)
                num_different_rows = torch.sum(differences).item()
                print(f"Number of rows updated by K-means: {num_different_rows}")
                
                # update the embeddings
                embedding_manager.update_all_chunks(sample_ids, updated_embeddings)
                embedding_manager.load_embeddings()
                
                # Check if the saved embeddings are the same as the updated embeddings
                # updated_embeddings_2 = embedding_manager.get_all_embeddings()[1]
                
                # unique_embeddings_2, _ = torch.unique(updated_embeddings_2, return_inverse=True, dim=0)
                # print(f"Unique embeddings loaded: {unique_embeddings_2.size(0)}")
                
                # # # Check if embeddings have been updated
                # differences = torch.any(updated_embeddings != updated_embeddings_2, dim=1)
                # num_different_rows = torch.sum(differences).item()
                # assert num_different_rows == 0, f"Embeddings have not been updated after clustering, {num_different_rows} rows are different"
                
                if cfg.control.save:
                    # Calculate the updated UMAP and KMeans clustering
                    umap_features_updated = clustering.get_umap(updated_embeddings)
                    umap_labels_updated, _ = clustering.get_kmeans(
                        umap_features_updated, n_clusters=n_clusters
                    )
                    
                    umap_features_np_updated = umap_features_updated.cpu().numpy()
                    umap_labels_np_updated = umap_labels_updated.cpu().numpy()

                    # Plot UMAP after clustering update
                    plot_umap(
                        umap_features_np_updated,
                        umap_labels_np_updated,
                        plot_dir,
                        f"{epoch}_kmupdate",
                        samples_to_track,
                    )
                
            elif epoch >= k_means_middle_epoch:
                print("##########No clustering##########")
                # Load embeddings
                embedding_manager.load_embeddings()
                sample_ids, label_embedding = embedding_manager.get_all_embeddings()
                unique_embeddings, _ = torch.unique(label_embedding, return_inverse=True, dim=0)
                print(f"Unique embeddings: {unique_embeddings.size(0)}")
            
        if cfg.control.test:
            if unique_embeddings is not None and epoch >= k_means_middle_epoch:
                print("##########Testing test dataset##########")
                inf_test_log = inference_test(model, tokenizer, test_dataloader, unique_embeddings, device, epoch, [1, 5, 10])
                logger_epoch["inference_test"] = inf_test_log
                wandb_run.log(inf_test_log)

        # Save model, epoch, optimizer, scheduler
        if cfg.control.save:
            # Save merge history
            folder_manager.save_model(model, checkpoint_dir, epoch)

        # Save logger per epoch
        logger.append(logger_epoch)

    # Save final model and merge history
    folder_manager.save_final_model(model, experiment_dir)
    # folder_manager.save_merge_history(embedding_manager.merge_history, experiment_dir)
    # feature_manager.close()
    
    # Clean cuda cache
    del (
        model,
        train_dataset,
        train_dataloader,
        criteria,
        optimizer,
        scheduler,
    )
    torch.cuda.empty_cache()

    return logger


@hydra.main(config_path="configs", config_name="flickr30k_mini", version_base=None)
def main(cfg):
    # Set seed
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize wandb
    project = cfg.wandb.project
    entity = cfg.wandb.entity
    tags = cfg.wandb.tags
    wandb.require("core")
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_run = wandb.init(project=project, entity=entity, tags=tags)
    
    # Save a copy of config file
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_dir = f"logs/{now}_{cfg.log_tag}"
    os.mkdir(logger_dir)
    OmegaConf.save(config=cfg, f=os.path.join(logger_dir, "config.yaml"))
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Run main function
    logger = run(cfg=cfg, logger_dir=logger_dir, wandb_run=wandb_run)
    json.dump(logger, open(os.path.join(logger_dir, "logger.json"), "w"))
    
    wandb.finish()


if __name__ == "__main__":
    main()
