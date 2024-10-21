import json
import os
import random
import warnings
from datetime import datetime
from turtle import update

import hydra
import numpy as np
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

import wandb

# Import local packages
from src.data.cdc_datamodule import CDC_test
from src.data.cdc_datamodule import CDC_train_preextract as CDC_train
from src.metric.loss import LabelContrastiveLoss
from src.models.cdc import CDC
from src.models.components.clustering import Clustering
from src.utils import (
    EmbeddingManager,
    FeatureManager,
    FolderManager,
    calculate_n_clusters_3,
    plot_umap,
    plot_umap_nooutlier,
    print_model_info,
)
from src.utils.inference import (
    extract_and_store_features,
    inference_test,
    inference_train,
    replace_with_most_different,
)

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg: DictConfig, **kwargs):
    model = kwargs["model"]
    train_dataloader = kwargs["train_dataloader"]
    criteria = kwargs["criteria"]
    optimizer = kwargs["optimizer"]
    epoch = kwargs["epoch"]
    scheduler = kwargs["scheduler"]
    embedding_manager = kwargs["embedding_manager"]
    update_label_embedding = kwargs["update_label_embedding"]
    high_lr = kwargs["high_lr"]
    log_interval = cfg.train.log_interval
    wandb_run = kwargs["wandb_run"]

    model.train()
    epoch_metrics = {"loss": 0.0, "other_metrics": {}}

    for batch_id, batch in enumerate(tqdm(train_dataloader)):
        img_emb, txt_emb, txt_full, label_embedding, sample_id = batch
        img_emb, txt_emb, txt_full, label_embedding = (
            img_emb.squeeze(0),
            txt_emb.squeeze(0),
            txt_full.squeeze(0),
            label_embedding.squeeze(0),
        )

        img_emb, txt_emb, txt_full = (
            img_emb.to(device),
            txt_emb.to(device),
            txt_full.to(device),
        )

        label_embedding = label_embedding.to(device).clone().detach().requires_grad_(True)
        label_embedding_cp = label_embedding.clone().detach()

        # Initialize optimizer for label_embedding
        # current_lr = optimizer.param_groups[0]["lr"] * 100
        # if high_lr:  # high learning rate for label
        #     current_lr = 1e-1
        # else:  # low learning rate for label
        #     current_lr = cfg.train.lr
        current_lr = 1e-5
        optimizer_label = torch.optim.AdamW(
            [label_embedding],
            lr=current_lr,
            weight_decay=cfg.train.weight_decay,
            betas=(cfg.train.betas[0], cfg.train.betas[1]),
        )

        comb_emb = model.module.combine(txt_emb, txt_full, label_embedding)

        label_embedding_neg = replace_with_most_different(
            label_embedding
        )  # Sample new label embeddings
        comb_emb_neg = model.module.combine(txt_emb, txt_full, label_embedding_neg)

        loss_dict = criteria(img_emb, txt_emb, comb_emb, comb_emb_neg)
        loss = loss_dict["total_loss"]
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
        if scheduler is not None:
            scheduler.step(epoch + batch_id / len(train_dataloader))
        if batch_id % log_interval == 0 or batch_id == len(train_dataloader) - 1:
            print(
                f"Epoch: {epoch}, Batch: {batch_id} / {len(train_dataloader)-1 }, Loss: {loss.item()}, Dynamic Scalar: {model.module.combiner.print_scalar()}"
            )

        # Wandb logger
        wandb_run.log(
            {
                "train/epoch": epoch,
                "train/total_loss": loss.item(),
                "train/comb_contrastive_loss": loss_dict["comb_contrastive_loss"].item(),
                "train/label_contrastive_loss": loss_dict["label_contrastive_loss"].item(),
                "train/diff_contrastive_loss": loss_dict["diff_contrastive_loss"].item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/dynamic_scalar": model.module.combiner.get_newest(),
            },
        )

        del (
            img_emb,
            txt_full,
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
    # Initialize Model
    model = CDC(
        clip_trainable=False,
        d_model=cfg.model.d_model,
        nhead=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        label_dim=cfg.model.label_dim,
    )
    model = nn.DataParallel(model)
    model.to(device)

    # Print model summary
    print_model_info(model)

    # preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Initialize FolderManager
    folder_manager = FolderManager(base_log_dir=cfg.dataset.log_path)

    # Initialize feature manager
    feature_manager = FeatureManager(cfg.dataset.extract_path, chunk_size=cfg.train.batch_size)

    # Initialize experiment
    experiment_dir, init_dir, plot_dir = folder_manager.initialize_experiment(cfg.log_tag)
    checkpoint_dir, logs_dir = folder_manager.create_directories(experiment_dir)

    if cfg.dataset.pre_extract:
        print("##########Extracting and storing features##########")
        sample_ids_list = extract_and_store_features(
            cfg.dataset.train_path,
            cfg.dataset.img_path,
            feature_manager,
            cfg.train.batch_size,
            model,
            processor,
            device,
            ratio=cfg.dataset.ratio,
        )
        torch.save(
            sample_ids_list,
            os.path.join(cfg.dataset.extract_path, "sample_ids_list.pt"),
        )
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
        annotations,
        embedding_dim=cfg.model.label_dim,
        chunk_size=cfg.train.batch_size,
        embeddings_dir=init_dir,
        sample_ids_list=sample_ids_list,
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

    # Setup criteria and optimizer and scheduler
    criteria = LabelContrastiveLoss(
        margin=0.1,
        margin_pos=0.5,
        margin_neg=0.1,
        label_weight=1,
        diff_weight=0.1,
        return_dict=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.betas[0], cfg.train.betas[1]),
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.train.warm_up, T_mult=1, eta_min=cfg.train.lr_min
    )

    # For training before testing
    optimizer_tmp = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.betas[0], cfg.train.betas[1]),
    )

    # Callbacks
    logger = []
    max_epoch = cfg.train.max_epochs
    cap_n_clusters = cfg.train_2.cap_n_clusters  # Cap number of clusters
    initial_n_clusters = min(
        abs(train_dataset.get_len() - cfg.train_2.initial_n_clusters), cap_n_clusters
    )  # Initial number of clusters
    first_stage_n = cfg.train_2.first_stage_n  # Number of clusters after first stage
    second_stage_n = cfg.train_2.second_stage_n  # Number of clusters after second stage
    k_means_start_epoch = cfg.train_2.k_means_start_epoch  # Start k-means clustering
    k_means_middle_epoch = cfg.train_2.k_means_middle_epoch  # Start slow k-means clustering
    k_means_end_epoch = cfg.train_2.k_means_end_epoch  # End k-means clustering
    update_label_embedding = True  # Update label embeddings during training
    high_lr = True

    assert (
        k_means_start_epoch <= k_means_end_epoch <= max_epoch
    ), f"Invalid epoch values for k-means clustering, {k_means_start_epoch}, {k_means_end_epoch}, {max_epoch}"
    assert (
        initial_n_clusters >= first_stage_n >= second_stage_n
    ), f"Invalid number of clusters, {initial_n_clusters}, {first_stage_n}, {second_stage_n}"

    n_clusters_list = calculate_n_clusters_3(
        initial_n_clusters,
        first_stage_n,
        second_stage_n,
        max_epoch,
        k_means_start_epoch,
        k_means_middle_epoch,
        k_means_end_epoch,
        decay_rate=0.75,
        interval=1,
    )

    # Start training
    for epoch in range(max_epoch):
        logger_epoch = {}
        logger_epoch["epoch"] = epoch
        unique_embeddings = None

        # Train
        if cfg.control.train:  # Network training
            print(f"##########Epoch {epoch}: Training##########")

            # # Create new directory for the current epoch
            if cfg.control.save_per_epoch is True:
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
                train_dataloader=train_dataloader,
                epoch=epoch,
                criteria=criteria,
                optimizer=optimizer,
                embedding_manager=embedding_manager,
                update_label_embedding=update_label_embedding,
                high_lr=high_lr,
                scheduler=scheduler,
                wandb_run=wandb_run,
            )
            scheduler.step()
            logger_epoch["train"] = train_epoch_log

            # Save epoch metrics
            folder_manager.save_metrics(train_epoch_log, logs_dir, epoch)

        if cfg.control.val:
            print("##########Testing train dataset##########")
            inf_train_log = inference_train(model, train_dataloader, device, epoch, [1, 5, 10])
            wandb_run.log(inf_train_log)
            logger_epoch["inference_train"] = inf_train_log

        if cfg.control.train_2:  # KMeans update
            n_clusters = n_clusters_list[epoch]  # Number of clusters for the current epoch
            # An adaptive alpha which minimum 0.1 and maximum 0.9, slide depends on k_means_middle_epoch - k_means_start_epoch
            alpha = 0.9 * (1 - (k_means_middle_epoch - epoch) / k_means_middle_epoch)

            # Perform clustering and update embeddings by merging
            if (
                k_means_start_epoch
                <= epoch
                < k_means_end_epoch
                # and n_clusters_list[max(epoch - 1, 0)] != n_clusters
            ):
                print(
                    f"##########Epoch {epoch}: Expected number of clusters: {n_clusters}##########"
                )

                # Load embeddings
                embedding_manager.load_embeddings()
                sample_ids, label_embedding = embedding_manager.get_all_embeddings()

                # Perform UMAP and clustering on unique embeddings
                print("##########Performing UMAP for computation##########")
                umap_features_high = clustering.get_umap(
                    label_embedding, n_components=cfg.train_2.umap_components
                )

                print("##########Performing Clustering##########")
                umap_labels, _ = clustering.get_hdbscan(umap_features_high, n_clusters=n_clusters)
                unique_umap_labels = torch.unique(umap_labels)
                print(f"Unique UMAP labels: {unique_umap_labels.size(0)}")

                # Map clustering results back to the original embeddings
                if epoch < k_means_middle_epoch:
                    update_noise = "ignore"
                    update_type = "soft"
                    print("##########Performing soft clustering update##########")
                else:
                    update_noise = "assign"
                    update_type = "hard"
                    high_lr = False
                    print("##########Performing hard clustering update##########")

                updated_embeddings, cluster_centers = clustering.hdbscan_update(
                    umap_labels=umap_labels,
                    original_embeddings=label_embedding,
                    update_type=update_type,
                    alpha=alpha,
                    update_noise=update_noise,
                )

                # Find unique embeddings
                unique_embeddings, _ = torch.unique(updated_embeddings, return_inverse=True, dim=0)
                print(f"Unique embeddings after clustering update: {unique_embeddings.size(0)}")

                # Check how many embeddings have been updated by k-means
                differences = torch.any(label_embedding != updated_embeddings, dim=1)
                num_different_rows = torch.sum(differences).item()
                print(f"Number of rows updated by Clustering: {num_different_rows}")

                print(f"Number of true cluster centers after update: {cluster_centers.size(0)}")

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
                    # Plot UMAP before clustering update

                    print("##########Performing UMAP for visualisation##########")
                    umap_features, updated_umap_features = clustering.get_and_predict_umap(
                        label_embedding, updated_embeddings
                    )

                    umap_features_np = umap_features.cpu().numpy()
                    umap_labels_np = umap_labels.cpu().numpy()
                    updated_umap_features_np = updated_umap_features.cpu().numpy()

                    path_umap = plot_umap(
                        umap_features_np,
                        umap_labels_np,
                        plot_dir,
                        epoch,
                        samples_to_track,
                    )

                    path_umap_after_update = plot_umap(
                        updated_umap_features_np,
                        umap_labels_np,
                        plot_dir,
                        f"{epoch}_after_update",
                        samples_to_track,
                    )

                    path_umap_nooutlier = plot_umap_nooutlier(
                        umap_features_np,
                        umap_labels_np,
                        plot_dir,
                        epoch,
                        samples_to_track,
                    )

                wandb_run.log(
                    {
                        "train_2/epoch": epoch,
                        "train_2/n_clusters": n_clusters,
                        "train_2/alpha": alpha,
                        "train_2/n_unique": unique_embeddings.size(0),
                        "train_2/n_clusters_center": cluster_centers.size(0),
                        # Log image by wandb
                        "train_2/umap": wandb.Image(path_umap),
                        "train_2/umap_after_update": wandb.Image(path_umap_after_update),
                        "train_2/umap_nooutlier": wandb.Image(path_umap_nooutlier),
                    }
                )

                # Save unique embeddings
                torch.save(
                    unique_embeddings[:50],
                    os.path.join(experiment_dir, "unique_embeddings.pt"),
                )

                if unique_embeddings.size(0) > 50:
                    unique_embeddings = cluster_centers

            elif epoch >= k_means_middle_epoch:
                print("##########No clustering##########")
                # Load embeddings
                embedding_manager.load_embeddings()
                sample_ids, label_embedding = embedding_manager.get_all_embeddings()
                unique_embeddings, _ = torch.unique(label_embedding, return_inverse=True, dim=0)
                print(f"Unique embeddings: {unique_embeddings.size(0)}")

        if cfg.control.test:
            if unique_embeddings is not None:  # and epoch >= k_means_middle_epoch:
                if update_label_embedding is True:
                    print("##########Adaptation before testing##########")
                    _ = train(
                        cfg,
                        model=model,
                        train_dataloader=train_dataloader,
                        epoch=epoch,
                        criteria=criteria,
                        optimizer=optimizer_tmp,
                        embedding_manager=embedding_manager,
                        update_label_embedding=False,
                        high_lr=False,
                        scheduler=None,
                        wandb_run=wandb_run,
                    )

                print("##########Testing test dataset##########")
                inf_test_log = inference_test(
                    model, processor, test_dataloader, unique_embeddings, epoch, device
                )
                logger_epoch["inference_test"] = inf_test_log
                wandb_run.log(inf_test_log)

        # # Save model, epoch, optimizer, scheduler
        # if cfg.control.save:
        #     # Save merge history
        #     folder_manager.save_model(model, checkpoint_dir, epoch)

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


@hydra.main(config_path="configs", config_name="redcaps", version_base=None)
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
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_run = wandb.init(project=project, entity=entity, tags=tags, config=config_dict)  # type: ignore

    # Save a copy of config file
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_dir = f"logs/{now}_{cfg.log_tag}"
    os.mkdir(logger_dir)
    OmegaConf.save(config=cfg, f=os.path.join(logger_dir, "config.yaml"))
    # Print config
    print(OmegaConf.to_yaml(cfg))
    # Save config to wandb

    # Run main function
    logger = run(cfg=cfg, logger_dir=logger_dir, wandb_run=wandb_run)
    json.dump(logger, open(os.path.join(logger_dir, "logger.json"), "w"))

    wandb.finish()


if __name__ == "__main__":
    main()
