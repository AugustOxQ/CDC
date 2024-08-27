import json
import os
from turtle import update
import warnings
from datetime import datetime
from altair import sample
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import random

import hydra
from sqlalchemy import all_
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
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
    
def inference_train(model, tokenizer, dataloader, device, epoch, Ks=[1, 5, 10], max_batches=10):
    # Read embeddings directly from the dataloader, compare with other mebeddings from the same batch
    model.eval()
    total_raw_better_count = 0
    total_shuffled_better_count = 0
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

            # # Test 3: Precision and Recall@K of cosine_sim_comb
            batch_size = cosine_sim_comb.shape[0]
            total_samples += batch_size
                        
            del img_emb, txt_emb, label_embedding, comb_emb, comb_emb_shuffled, label_embedding_shuffled
            
            torch.cuda.empty_cache()
            
    # # Compute precision@K
    # for k in Ks:
    #     total_precisions[k] /= (total_samples * k)
    
    # Calculate percentage of better label embeddings
    raw_better_percentage = total_raw_better_count / total_samples * 100
    shuffled_better_percentage = total_shuffled_better_count / total_samples * 100
    
    print(f"Epoch {epoch}: Combined embeddings better than raw embeddings: {raw_better_percentage:.2f}%")
    print(f"Epoch {epoch}: Combined embeddings better than shuffled embeddings: {shuffled_better_percentage:.2f}%")
    # for k in Ks:
    #     print(f'Epoch {epoch}: Precision@{k}: {total_precisions[k] * 100:.2f}%')

    return {
        "val/raw_better_percentage": raw_better_percentage,
        "val/shuffled_better_percentage": shuffled_better_percentage,
        # "precisions": total_precisions
    }


def inference_test(model, tokenizer, dataloader, label_embeddings, device, epoch, Ks=[1, 5, 10]):
    # Load unique label embeddings up to 50
    label_embeddings = label_embeddings[:50]
    
    total_samples = 0
    total_better_count = 0
    total_improvement = 0.0
    
    all_img_emb = []
    all_txt_emb = []
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
            
            # Convert PyTorch tensors to NumPy arrays
            img_emb_np = img_emb.cpu().numpy()
            txt_emb_np = txt_emb.cpu().numpy()

            best_cosine_sim = np.ones((batch_size, captions_per_image)) * -1 # Initialize to -1
            best_comb_emb = torch.zeros((batch_size, captions_per_image, label_embeddings.size(1)))

            for label_embedding in label_embeddings:
                label_embedding = label_embedding.to(device)
                comb_emb = model.combine(txt_emb, label_embedding.unsqueeze(0).expand(txt_emb.size(0), -1))

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
                            
                    # max_cosine_sim_comb = np.max(cosine_sim_comb[i * 5 + j])

            # Compare best cosine similarity with raw cosine similarity
            cosine_sim_raw = np.array([cosine_similarity(img_emb_np[i:i+1], txt_emb_np[i*5:(i+1)*5]).flatten() for i in range(batch_size)])
            
            total_better_count += np.sum(best_cosine_sim > cosine_sim_raw)
            total_samples += batch_size
            
            # Calculate improvement
            improvement = (best_cosine_sim - cosine_sim_raw) / cosine_sim_raw
            total_improvement += np.sum(improvement)
            
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
    print(f"Inference Test - Percentage of better cosine similarity: {better_percentage:.2f}%")
    print(f"Inference Test - Average improvement over raw cosine similarity: {avg_improvement:.2f}%")
    
    metrics_raw = evalrank(all_img_emb, all_txt_emb, text_to_image_map, image_to_text_map, "raw")
    
    metrics_comb = evalrank(all_img_emb, all_best_comb_emb, text_to_image_map, image_to_text_map, "comb")
    
    metrics_basic = {
        "test/better_percentage": better_percentage,
        "test/avg_improvement": avg_improvement,
    }
    
    metrics_total = {**metrics_basic, **metrics_raw, **metrics_comb}
    
    return metrics_total, all_img_emb, all_txt_emb, all_best_comb_emb
    
    # return {
    #     "test/better_percentage": better_percentage,
    #     "test/avg_improvement": avg_improvement,
    #     # "test/recalls_raw": recalls_raw,
    #     # "test/recalls_comb": recalls_comb
    # }, all_img_emb, all_txt_emb, all_best_comb_emb


def train(cfg: DictConfig, **kwargs):
    model = kwargs["model"]
    tokenizer = kwargs["tokenizer"]
    train_dataloader = kwargs["train_dataloader"]
    criteria = kwargs["criteria"]
    optimizer = kwargs["optimizer"]
    epoch = kwargs["epoch"]
    scheduler = kwargs["scheduler"]
    embedding_manager = kwargs["embedding_manager"]
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
        
        # sample_id = [int(s) for s in sample_id]

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
        optimizer_label.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_label.step()
        
        # Check if label_embedding is updated
        diff = torch.sum(label_embedding - label_embedding_cp)
        assert diff != 0, "Label embedding is not updated"
        
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
        annotations, embedding_dim=512, chunk_size=cfg.train.batch_size, hdf5_dir=init_dir, sample_ids_list=sample_ids_list
    )

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
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train.T_max, eta_min=cfg.train.lr_min
    )

    # Callbacks
    logger = []
    max_epoch = cfg.train.max_epochs
    
    initial_n_clusters = train_dataset.get_len() - cfg.train_2.initial_n_clusters
    first_stage_n = cfg.train_2.first_stage_n
    second_stage_n = cfg.train_2.second_stage_n
    k_means_start_epoch = cfg.train_2.k_means_start_epoch
    k_means_slow_epoch = cfg.train_2.k_means_slow_epoch
    
    assert k_means_start_epoch < k_means_slow_epoch < max_epoch, "Invalid epoch values for k-means clustering"
    
    assert initial_n_clusters > first_stage_n > second_stage_n, "Invalid number of clusters"

    # Start training
    for epoch in range(max_epoch):
        logger_epoch = {}
        logger_epoch["epoch"] = epoch
        
        unique_embeddings = None
        
        # Train
        if cfg.control.train: # Network training
            print(f"##########Epoch {epoch}: Training##########")
            
            # Create new directory for the current epoch
            new_hdf5_dir = folder_manager.create_epoch_folder(epoch)
            embedding_manager.hdf5_dir = new_hdf5_dir
            embedding_manager.save_embeddings_to_new_folder(new_hdf5_dir)
            embedding_manager.load_embeddings()
            
            train_epoch_log = train(
                cfg,
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                epoch=epoch,
                criteria=criteria,
                optimizer=optimizer,
                embedding_manager=embedding_manager,
                scheduler=scheduler,
                wandb_run=wandb_run,
            )
            scheduler.step()
            logger_epoch["train"] = train_epoch_log
            
            # Save epoch metrics
            folder_manager.save_metrics(train_epoch_log, logs_dir, epoch)
    
        if cfg.control.train_2: # KMeans update
            n_clusters = calculate_n_clusters(initial_n_clusters, first_stage_n, second_stage_n, epoch, k_means_start_epoch, k_means_slow_epoch)
            wandb_run.log({"train/n_clusters": n_clusters})
            
            if n_clusters == 0:
                print("##########No clustering performed##########")
                
            else:
                # Only perform clustering if n_clusters is not 0
                print(f"##########Epoch {epoch}: Number of clusters to be cluster: {n_clusters}##########")
            
                # Create new directory for the current epoch
                new_hdf5_dir = folder_manager.create_epoch_folder(f"{epoch}_kmupdate")
                embedding_manager.hdf5_dir = new_hdf5_dir
                embedding_manager.save_embeddings_to_new_folder(new_hdf5_dir)
                embedding_manager.load_embeddings()

                # Perform clustering and merge embeddings using proxy embeddings
                label_embedding = embedding_manager.get_all_embeddings()[1]

                # Perform UMAP and clustering on unique embeddings
                print("##########Performing UMAP##########")
                umap_features = clustering.get_umap(label_embedding)
                
                print("##########Performing KMeans##########")
                umap_labels, centers = clustering.get_kmeans(
                umap_features, n_clusters=n_clusters
                )
                umap_features_np = umap_features.cpu().numpy()
                umap_labels_np = umap_labels.cpu().numpy()

                # Plot UMAP before clustering update
                plot_umap(
                    umap_features_np, umap_labels_np, plot_dir, epoch, samples_to_track
                )
                print("##########Performing clustering update##########")
                
                # Map clustering results back to the original embeddings
                updated_embeddings = clustering.kmeans_update(
                    umap_labels=umap_labels,
                    centers = centers,
                    original_embeddings=label_embedding,
                    update_type='hard',
                    alpha=0.1
                )
                
                # Find unique embeddings
                unique_embeddings, _ = torch.unique(
                    updated_embeddings, return_inverse=True, dim=0
                )
                print(f"Unique embeddings after clustering update: {unique_embeddings.size(0)}")
                
                if unique_embeddings.size(0) <= cfg.eval.max_clusters:
                    torch.save(unique_embeddings, os.path.join(experiment_dir, f"unique_embeddings_{epoch}.pt"))
                
                # Check if embeddings have been updated
                differences = torch.any(label_embedding != updated_embeddings, dim=1)
                num_different_rows = torch.sum(differences).item()
                print(num_different_rows)
                
                # update the embeddings
                embedding_manager.update_all_chunks(updated_embeddings)
                embedding_manager.load_embeddings()
                
                
                #TODO: Double check this
                # updated_embeddings = embedding_manager.get_all_embeddings()[1]
                
                # # Check if embeddings have been updated
                # differences = torch.any(updated_embeddings != label_embedding, dim=1)
                # num_different_rows = torch.sum(differences).item()
                # print(f"Number of different rows after update: {num_different_rows}")
                
                # Calculate the updated UMAP and KMeans clustering
                umap_features_updated = clustering.get_umap(updated_embeddings)
                umap_labels_updated, centers_updated = clustering.get_kmeans(
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
            
        if cfg.control.val:
            print("##########Testing train dataset##########")
            inf_train_log = inference_train(model, tokenizer, train_dataloader, device, epoch, [1, 5, 10])
            wandb_run.log(inf_train_log)
            logger_epoch["inference_train"] = inf_train_log
            
        if cfg.control.test:      
            if unique_embeddings is not None:      
                # if unique_embeddings.size(0) <= cfg.eval.max_clusters:
                print("##########Testing test dataset##########")
                print(f"Unique embeddings: {unique_embeddings.size(0)}")
                inf_test_log, all_img_emb, all_txt_emb, all_best_comb_emb = inference_test(model, tokenizer, test_dataloader, unique_embeddings, device, epoch, [1, 5, 10])
                logger_epoch["inference_test"] = inf_test_log
                wandb_run.log(inf_test_log)
                
                # Save embeddings for visualization
                if cfg.control.save:
                    torch.save(all_img_emb, os.path.join(experiment_dir,f"all_img_emb_{epoch}.pt"))
                    torch.save(all_txt_emb, os.path.join(experiment_dir, f"all_txt_emb_{epoch}.pt"))
                    torch.save(all_best_comb_emb, os.path.join(experiment_dir, f"all_best_comb_emb_{epoch}.pt"))

            
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


@hydra.main(config_path="configs", config_name="flickr30k", version_base=None)
def main(cfg):
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    project = cfg.wandb.project
    entity = cfg.wandb.entity
    tags = cfg.wandb.tags

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Save config
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_dir = f"logs/{now}_{cfg.log_tag}"
    os.mkdir(logger_dir)
    OmegaConf.save(config=cfg, f=os.path.join(logger_dir, "config.yaml"))
    
    wandb.require("core")
    wandb_run = wandb.init(project=project, entity=entity, tags=tags)
    logger = run(cfg=cfg, logger_dir=logger_dir, wandb_run=wandb_run)
    json.dump(logger, open(os.path.join(logger_dir, "logger.json"), "w"))
    
    wandb.finish()


if __name__ == "__main__":
    main()
