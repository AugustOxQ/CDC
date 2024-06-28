import json
import os
import warnings
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

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
from src.data.imp_datamodule import CDC_train, CDC_test, EmbeddingManager, FolderManager
from src.metric.loss import CosineLoss, MeanSquareLoss
from src.models.cdc import CDC
from src.models.components.clustering import Clustering
from src.utils import EmbeddingManager, FolderManager, evalrank

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_n_clusters(initial_n_clusters, final_n_clusters, epoch, total_epochs):
    # Sigmoid function to control the rate of decrease
    x = np.linspace(-6, 6, total_epochs)  # Spread over a range to adjust the curve
    sigmoid = 1 / (1 + np.exp(-x))

    # Normalize to get values between 0 and 1
    normalized_sigmoid = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())

    # Invert and scale to get the number of clusters
    n_clusters = initial_n_clusters - normalized_sigmoid * (
        initial_n_clusters - final_n_clusters
    )

    # Return the number of clusters for the current epoch
    return int(n_clusters[epoch])


def plot_umap(umap_features_np, umap_labels, plot_dir, epoch, samples_to_track=[]):
    # Plot UMAP before clustering update
    fig = plt.figure(figsize=(16, 16))
    plt.scatter(umap_features_np[:, 0], umap_features_np[:, 1], c=umap_labels, s=0.1)

    # Highlight and annotate the tracked samples
    for sample_idx in samples_to_track:
        x, y = umap_features_np[sample_idx, :]
        plt.scatter(x, y, c="red", s=150, edgecolors="k")  # Highlight the sample
        plt.text(
            x, y, f"Sample {sample_idx}", fontsize=24, color="black"
        )  # Annotate the sample

    # Add the number of umap_labels to the plot as title
    plt.title(f"UMAP with {len(umap_labels)} clusters")
    
    plt.colorbar()
    # output the figure
    plt.savefig(os.path.join(plot_dir, f"umap_{epoch}.png"))
    plt.close(fig)


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

    initial_n_clusters = len(train_dataset) - 10
    final_n_clusters = 5
    num_epochs = cfg.train.max_epochs

    # Start training
    max_epoch = cfg.train.max_epochs
    for epoch in range(max_epoch):
        logger_epoch = {}
        logger_epoch["epoch"] = epoch

        # Train
        if cfg.control.train:
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

        if cfg.control.val:

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
            umap_features = clustering.get_umap(unique_embeddings)
            n_clusters = calculate_n_clusters(
                initial_n_clusters, final_n_clusters, epoch, num_epochs
            )
            umap_labels, centers = clustering.get_kmeans(
                umap_features, n_clusters=n_clusters
            )

            umap_features_np = umap_features.cpu().numpy()
            umap_labels_np = umap_labels.cpu().numpy()

            # Plot UMAP before clustering update
            plot_umap(
                umap_features_np, umap_labels_np, plot_dir, epoch, samples_to_track
            )

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
