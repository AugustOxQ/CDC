import json
import os
import warnings
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

import hydra
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
from src.data.cdc_datamodule import CDC_train, CDC_test
from src.metric.loss import CosineLoss, MeanSquareLoss
from src.models.cdc import CDC
from src.models.components.clustering import Clustering
from src.utils import EmbeddingManager, FolderManager, evalrank, calculate_n_clusters, plot_umap

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
def k_means_stop_condition():
    #TODO: Keep track of the diversity of the k-means centroids by calculating the mean of pairwise distances between the centroids. This can be an automatic way of deciding when to stop the clustering process.
    
    pass
    
    
def inference_train(model, tokenizer, dataloader, device, epoch, Ks=[1, 5, 10], max_batches=5):
    # Read embeddings directly from the dataloader, compare with other mebeddings from the same batch
    model.eval()
    total_raw_better_count = 0
    total_shuffled_better_count = 0
    total_samples = 0
    total_precisions = {k: 0.0 for k in Ks}

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(dataloader)):
            
            # Limit the number of batches to process
            if batch_id >= max_batches:
                break
            
            image, raw_text, label_embedding, sample_id = batch
            image_input = image.to(device)
            text_input = tokenizer(
                raw_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            label_embedding = label_embedding.to(device)
            label_embedding_shuffled = label_embedding[torch.randperm(label_embedding.size(0))]
            del image, raw_text

            img_emb, txt_emb, lbl_emb, comb_emb = model.forward(
                image_input, text_input, label_embedding
            )
            
            comb_emb_shuffled = model.combine(
                txt_emb, label_embedding_shuffled
            )
            
            # Calculate cosine similarity within batch
            # Calculate cosine similarity between image and text embeddings
            cosine_sim_raw = cosine_similarity(
                img_emb.cpu().numpy(),
                txt_emb.cpu().numpy()
            )
            
            # Calculate cosine similarity between image and combined embeddings
            cosine_sim_comb = cosine_similarity(
                img_emb.cpu().numpy(),
                comb_emb.cpu().numpy()
            )
            
            # Calculate cosine similarity between image and combined embeddings (shuffled)
            cosine_sim_comb_shuffled = cosine_similarity(
                img_emb.cpu().numpy(),
                comb_emb_shuffled.cpu().numpy()
            )
            
            # Test 1: Whether cosine_sim_comb is greater than cosine_sim_raw
            comparison_raw = cosine_sim_comb >= cosine_sim_raw
            raw_better_count = np.sum(comparison_raw)
            total_raw_better_count += raw_better_count

            # Test 2: Whether cosine_sim_comb is greater than cosine_sim_comb_shuffled
            comparison_shuffled = cosine_sim_comb >= cosine_sim_comb_shuffled
            shuffled_better_count = np.sum(comparison_shuffled)
            total_shuffled_better_count += shuffled_better_count

            # Test 3: Precision and Recall@K of cosine_sim_comb
            batch_size = cosine_sim_comb.shape[0]
            total_samples += batch_size

            for i in range(batch_size):
                top_k_indices = np.argsort(-cosine_sim_comb[i])[:max(Ks)]
                for k in Ks:
                    if i in top_k_indices[:k]:
                        total_precisions[k] += 1
                        
            del img_emb, txt_emb, lbl_emb, comb_emb, comb_emb_shuffled, image_input, text_input, label_embedding, label_embedding_shuffled
            
            torch.cuda.empty_cache()
            
    # Compute precision@K
    for k in Ks:
        total_precisions[k] /= (total_samples * k)
    
    # Calculate percentage of better label embeddings
    raw_better_percentage = total_raw_better_count / (total_samples * cosine_sim_comb.shape[1]) * 100
    shuffled_better_percentage = total_shuffled_better_count / (total_samples * cosine_sim_comb.shape[1]) * 100
    
    print(f"Epoch {epoch}: Combined embeddings better than raw embeddings: {raw_better_percentage:.2f}%")
    print(f"Epoch {epoch}: Combined embeddings better than shuffled embeddings: {shuffled_better_percentage:.2f}%")
    for k in Ks:
        print(f'Epoch {epoch}: Precision@{k}: {total_precisions[k] * 100:.2f}%')

    return {
        "raw_better_percentage": raw_better_percentage,
        "shuffled_better_percentage": shuffled_better_percentage,
        "precisions": total_precisions
    }


def inference_test(model, tokenizer, dataloader, embedding_manager, device, epoch, Ks=[1, 5, 10]):
    # Load unique label embeddings
    unique_labels = list(embedding_manager.index_mapping.keys())
    label_embeddings = torch.stack([embedding_manager.get_embedding(sample_id) for sample_id in unique_labels]).to(device)
    
    model.eval()
    total_samples = 0
    total_better_count = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(dataloader)):
            image, raw_text = batch
            image_input = image.to(device)
            batch_size = image_input.size(0)
            num_texts = 5 if len(raw_text[0]) == 5 else 1
            tmp_text_input = []
            
            for b in range(batch_size):
                for i in range(num_texts):
                    tmp_text_input.append(raw_text[i][b])
                    
            text_input = tokenizer(
                tmp_text_input,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(device)

            img_emb, txt_emb = model.encode_img_txt(image_input, text_input)

            comb_emb_list = []
            for label_embedding in label_embeddings:
                comb_emb = model.comb(txt_emb, label_embedding.unsqueeze(0).expand(len(text_input), -1))
                comb_emb_list.append(comb_emb)
            
            comb_emb_list = torch.cat(comb_emb_list, dim=1)

            # Calculate cosine similarity within batch
            cosine_sim_comb = cosine_similarity(
                img_emb.cpu().numpy(),
                comb_emb_list.cpu().numpy()
            )

            # Compare the highest cosine similarity
            for b in range(batch_size):
                txt_emb = txt_emb[b*num_texts:(b+1)*num_texts]
                cosine_sim_raw = cosine_similarity(
                    img_emb[b].unsqueeze(0).cpu().numpy(),
                    txt_emb.cpu().numpy()
                )
                raw_better_count = np.sum(np.max(cosine_sim_comb[b].reshape(num_texts, -1), axis=1) > np.max(cosine_sim_raw, axis=1))
                total_better_count += raw_better_count
                
            total_samples += batch_size
            
            del img_emb, txt_emb, comb_emb_list, image_input, text_input
            
    # Compute and print aggregated results
    better_percentage = total_better_count / total_samples * 100

    print(f"Epoch {epoch}: Combined embeddings better than raw embeddings: {better_percentage:.2f}%")

    return {
        "better_percentage": better_percentage
    }

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

    model.train()
    epoch_metrics = {
        "loss": 0.0,
        "other_metrics": {}
    }
    
    for batch_id, batch in enumerate(tqdm(train_dataloader)):
        image, raw_text, label_embedding, sample_id = batch

        image_input = image.to(device)
        text_input = tokenizer(
            raw_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(device)

        label_embedding = nn.Parameter(label_embedding, requires_grad=True)
        label_embedding = label_embedding.to(device)
        del image, raw_text

        img_emb, txt_emb, lbl_emb, comb_emb = model.forward(
            image_input, text_input, label_embedding
        )
        loss = criteria(img_emb, comb_emb)
        epoch_metrics["loss"] += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save updated embeddings
        for ip, le in zip(sample_id, lbl_emb):
            embedding_manager.update_embedding(ip, le.data)

        # Log
        scheduler.step(epoch + batch_id / len(train_dataloader))
        if batch_id % log_interval == 0 or batch_id == len(train_dataloader) - 1:
            print(
                f"Epoch: {epoch}, Batch: {batch_id} / {len(train_dataloader)-1 }, Loss: {loss.item()}"
            )

        del (
            image_input,
            text_input,
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

    model = CDC().to(device)
    preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize FolderManager
    folder_manager = FolderManager(base_log_dir=cfg.dataset.log_path)

    # Initialize experiment
    experiment_dir, init_dir, plot_dir = folder_manager.initialize_experiment(cfg.log_tag)
    checkpoint_dir, logs_dir = folder_manager.create_directories(experiment_dir)

    # Initialize embedding manager
    annotations = json.load(open(cfg.dataset.train_path))
    annotations = annotations[: int(len(annotations) * cfg.dataset.ratio)]
    embedding_manager = EmbeddingManager(
        annotations, embedding_dim=512, chunk_size=10000, hdf5_dir=init_dir
    )

    # Samples to track
    samples_to_track = [0, 1, 2, 3, 4]  # Indices of the samples to track

    # Initialize clustering
    clustering = Clustering(embedding_manager)

    # Create Train and Test dataloader
    train_dataset = CDC_train(
        annotation_path=cfg.dataset.train_path,
        image_path=cfg.dataset.img_path,
        preprocess=preprocess,
        embedding_manager=embedding_manager,
        ratio=cfg.dataset.ratio,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        # TODO: Fix num_workers, it is causing error. Which is a multithreading issue
        # num_workers=cfg.train.num_workers,
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
        # num_workers=cfg.train.num_workers,
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
    
    initial_n_clusters = len(train_dataset) - cfg.train_2.initial_n_clusters
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

        # Train
        if cfg.control.train: # Network training
            print(f"##########Epoch {epoch}: Training##########")
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
            )
            scheduler.step()
            logger_epoch["train"] = train_epoch_log
            
            # Save epoch metrics
            folder_manager.save_metrics(train_epoch_log, logs_dir, epoch)

            # Create new directory for the current epoch
            new_hdf5_dir = folder_manager.create_epoch_folder(epoch)
            embedding_manager.save_embeddings_to_new_folder(new_hdf5_dir)

            # Update the embedding manager's hdf5_dir to the new directory
            embedding_manager.hdf5_dir = new_hdf5_dir

        if cfg.control.train_2: # KMeans update
            n_clusters = calculate_n_clusters(initial_n_clusters, first_stage_n, second_stage_n, epoch, k_means_start_epoch, k_means_slow_epoch)
            
            if n_clusters == 0:
                print("##########No clustering performed##########")
                
            else:
                # Only perform clustering if n_clusters is not 0
                print(f"##########Epoch {epoch}: Number of clusters: {n_clusters}##########")
            
                # Create new directory for the current epoch
                new_hdf5_dir = folder_manager.create_epoch_folder(f"{epoch}_kmupdate")
                embedding_manager.save_embeddings_to_new_folder(new_hdf5_dir)

                # Update the embedding manager's hdf5_dir to the new directory
                embedding_manager.hdf5_dir = new_hdf5_dir

                # Perform clustering and merge embeddings using proxy embeddings
                label_embedding = torch.stack(
                    [
                        embedding_manager.get_proxy_embedding(sample_id)
                        for sample_id in range(len(train_dataset))
                    ]
                )

                # Identify unique embeddings and their original indices
                unique_embeddings, inverse_indices = torch.unique(
                    label_embedding, return_inverse=True, dim=0
                )

                # Perform UMAP and clustering on unique embeddings
                print("##########Performing UMAP##########")
                umap_features = clustering.get_umap(unique_embeddings)
                
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
                mapped_labels = umap_labels[inverse_indices]
                clustering.merge_embeddings(
                    mapped_labels,
                    centers,
                    label_embedding,
                )
                
                updated_embeddings = torch.stack(
                    [
                        embedding_manager.get_embedding(sample_id)
                            for sample_id in range(len(train_dataset))
                    ]
                )
                
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
            logger_epoch["inference_train"] = inf_train_log
            
            
        if cfg.control.test:
            if len(embedding_manager.index_mapping) <= cfg.eval.max_clusters:
                print("##########Testing test dataset##########")
                inf_test_log = inference_test(model, tokenizer, test_dataloader, embedding_manager, device, epoch, [1, 5, 10])
                logger_epoch["inference_test"] = inf_test_log

            
        # Save model, epoch, optimizer, scheduler
        if cfg.control.save:
            # Save merge history
            folder_manager.save_model(model, checkpoint_dir, epoch)

        # Save logger per epoch
        logger.append(logger_epoch)

    # Save final model and merge history
    folder_manager.save_final_model(model, experiment_dir)
    folder_manager.save_merge_history(embedding_manager.merge_history, experiment_dir)
    
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

    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Save config
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_dir = f"logs/{now}_{cfg.log_tag}"
    os.mkdir(logger_dir)
    OmegaConf.save(config=cfg, f=os.path.join(logger_dir, "config.yaml"))
    logger = run(cfg=cfg, logger_dir=logger_dir)
    json.dump(logger, open(os.path.join(logger_dir, "logger.json"), "w"))


if __name__ == "__main__":
    main()
