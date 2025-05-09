import json
import math
import os
import random
import warnings
from datetime import datetime
from enum import unique
from turtle import update

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from torch import nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
)
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
    diversity_loss,
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

WARM_UP_EPOCH = 0
K_means_gap = 1
USE_TEMPLATE = True


def annealing_rewarmup_schedule(
    epoch: int,
    T_max: int = 10,
    initial_lr: float = 1e-3,
    lr_min: float = 1e-9,
):
    # Adjusted epoch count after warmup
    t = epoch / T_max
    cosine_decay = 0.5 * (1 + math.cos(math.pi * t / T_max))
    min_scale = lr_min / initial_lr
    return min_scale + (1 - min_scale) * cosine_decay


def train(cfg: DictConfig, **kwargs):
    model = kwargs["model"]
    train_dataloader = kwargs["train_dataloader"]
    criteria = kwargs["criteria"]
    optimizer = kwargs["optimizer"]
    epoch = kwargs["epoch"]
    scheduler = kwargs["scheduler"]
    embedding_manager = kwargs["embedding_manager"]
    log_interval = cfg.train.log_interval
    wandb_run = kwargs["wandb_run"]

    model.train()
    epoch_metrics = {"loss": 0.0, "other_metrics": {}}

    if epoch == 0 and not USE_TEMPLATE:
        # Initialize label embedding
        w_t = model.module.combiner.label_proj_layer.weight.data.clone().detach().to(device)
        print("##########Initializing label embedding to compensate for the image##########")
        for batch_id, batch in enumerate(tqdm(train_dataloader)):
            img_emb, txt_emb, _, label_embedding, sample_id = batch
            img_emb, txt_emb, label_embedding = (
                img_emb.squeeze(0),
                txt_emb.squeeze(0),
                label_embedding.squeeze(0),
            )

            img_emb, txt_emb = (
                img_emb.to(device, non_blocking=True),
                txt_emb.to(device, non_blocking=True),
            )

            label_embedding_init = (img_emb - txt_emb) @ w_t
            embedding_manager.update_chunk_embeddings(batch_id, sample_id, label_embedding_init)

            if batch_id % log_interval == 0 or batch_id == len(train_dataloader) - 1:
                label_move_distance = torch.norm(label_embedding, p=2, dim=1).mean()
                print(
                    f"Epoch: {epoch}, Batch: {batch_id} / {len(train_dataloader)-1 }, Label Move Distance: {label_move_distance:.3f}"
                )

    for batch_id, batch in enumerate(tqdm(train_dataloader)):
        img_emb, txt_emb, txt_full, label_embedding, sample_id = batch
        img_emb, txt_emb, txt_full, label_embedding = (
            img_emb.squeeze(0),
            txt_emb.squeeze(0),
            txt_full.squeeze(0),
            label_embedding.squeeze(0),
        )

        img_emb, txt_emb, txt_full = (
            img_emb.to(device, non_blocking=True),
            txt_emb.to(device, non_blocking=True),
            txt_full.to(device, non_blocking=True),
        )

        label_embedding = (
            label_embedding.to(device, non_blocking=True).clone().detach().requires_grad_(True)
        )
        label_embedding_cp = label_embedding.clone().detach()

        base_lr = optimizer.get_last_lr()
        optimizer.add_param_group(
            {"params": [label_embedding], "lr": base_lr * 5e4}
        )  # Add label embedding to optimizer

        comb_emb, label_embedding_proj = model.module.combine(
            txt_emb, txt_full, label_embedding, epoch=epoch, return_label_proj=True
        )
        label_embedding_neg = replace_with_most_different(label_embedding)

        comb_emb_neg = model.module.combine(
            txt_emb, txt_full, label_embedding_neg, epoch=epoch, return_label_proj=False
        )

        loss_dict = criteria(
            img_emb, txt_emb, comb_emb, comb_emb_neg, label_embedding, label_embedding_proj
        )
        if epoch < WARM_UP_EPOCH:
            loss = loss_dict["early_loss"]
        else:
            loss = loss_dict["total_loss"]

        epoch_metrics["loss"] += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer.param_groups.pop()  # Popout label embedding from optimizer

        # # Check if label_embedding is updated
        # diff = torch.sum(label_embedding - label_embedding_cp)
        # if update_label_embedding:
        #     assert diff != 0, "Label embedding should be updated after backward pass"
        # else:
        #     assert (
        #         diff == 0
        #     ), "Label embedding should not be updated after backward pass"

        # if update_label_embedding:
        embedding_manager.update_chunk_embeddings(batch_id, sample_id, label_embedding)

        # Log
        if batch_id % log_interval == 0 or batch_id == len(train_dataloader) - 1:
            label_move_distance = torch.norm(
                label_embedding - label_embedding_cp, p=2, dim=1
            ).mean()
            emb1_norm = F.normalize(comb_emb, p=2, dim=1)
            emb2_norm = F.normalize(comb_emb_neg, p=2, dim=1)
            sim_pos_neg = emb1_norm @ emb2_norm.T
            triu_indices = torch.triu_indices(sim_pos_neg.size(0), sim_pos_neg.size(1), offset=1)
            upper_vals = sim_pos_neg[triu_indices[0], triu_indices[1]]
            pos_neg_diff = torch.mean(upper_vals).item()
            print(
                f"Epoch: {epoch}, Batch: {batch_id} / {len(train_dataloader)-1 }, Loss: {loss.item():.3f}, Dynamic Scalar: {model.module.combiner.print_scalar()}, Cosine Similarity (pos-neg): {pos_neg_diff:.4f}, Label Move Distance: {label_move_distance:.7f}"
            )

        wandb_run.log(
            {
                "train/epoch": epoch,
                "train/total_loss": loss_dict["total_loss"].item(),
                "train/loss_improve": loss_dict["loss_improve"].item(),
                "train/loss_neg": loss_dict["loss_neg"].item(),
                "train/label_change_loss": loss_dict["loss_label_change"].item(),
                "train/text_preserve_loss": loss_dict["loss_preserve"].item(),
                "train/loss_angular": loss_dict["loss_angular"].item(),
                "train/loss_pull_away": loss_dict["loss_pull_away"].item(),
                "train/boundary_loss": loss_dict["loss_boundary"].item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/diversity": pos_neg_diff,
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
    print_model_info(model.module.combiner)

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
        sample_ids_list = torch.load(
            os.path.join(cfg.dataset.extract_path, "sample_ids_list.pt"),
            weights_only=False,
        )
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
        use_template=USE_TEMPLATE,
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
        pin_memory=True,
    )

    test_dataset = CDC_test(
        annotation_path=cfg.dataset.test_path,
        image_path=cfg.dataset.img_path_test,
        processor=processor,
        ratio=0.2 if "redcaps" in cfg.dataset.test_path else 1,
    )

    print(f"Test dataset size: {len(test_dataset)}")  # 0.2 for redcaps, 1 for flickr30k

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # Setup criteria and optimizer and scheduler
    criteria = LabelContrastiveLoss(
        margin=0.3,
        lambda_pos=1.0,
        lambda_neg=1.0,
        lambda_labelchange=0.1,
        lambda_preserve=0.1,
        return_dict=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(cfg.train.betas[0], cfg.train.betas[1]),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.max_epochs,
        eta_min=cfg.train.lr_min,
        last_epoch=-1,
    )

    # scheduler = LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda epoch: annealing_rewarmup_schedule(
    #         epoch,
    #         T_max=cfg.train.warm_up,
    #         initial_lr=cfg.train.lr,
    #         lr_min=cfg.train.lr_min,
    #     ),
    # )

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
    k_means_middle_epoch = (
        cfg.train_2.k_means_middle_epoch
    )  # Start slow k-means clustering (hard update)
    k_means_end_epoch = cfg.train_2.k_means_end_epoch  # End k-means clustering
    alpha_upper = cfg.train_2.alpha_upper  # Upper bound for alpha
    update_label_embedding = True  # Update label embeddings during training

    assert (
        k_means_start_epoch <= k_means_end_epoch  # <= max_epoch
    ), f"Invalid epoch values for k-means clustering, {k_means_start_epoch}, {k_means_end_epoch}, {max_epoch}"
    # assert (
    #     initial_n_clusters >= first_stage_n >= second_stage_n
    # ), f"Invalid number of clusters, {initial_n_clusters}, {first_stage_n}, {second_stage_n}" #TODO: This is not necessary for HDBSCAN

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

    unique_embeddings = None
    # Start training
    for epoch in range(max_epoch):
        logger_epoch = {}
        logger_epoch["epoch"] = epoch

        # Train
        if cfg.control.train:  # Network training
            print(f"##########Epoch {epoch}: Training##########")

            # # # Create new directory for the current epoch
            # if cfg.control.save_per_epoch is True:
            #     new_embeddings_dir = folder_manager.create_epoch_folder(epoch)
            #     embedding_manager.embeddings_dir = new_embeddings_dir
            #     embedding_manager.save_embeddings_to_new_folder(new_embeddings_dir)
            # embedding_manager.load_embeddings()

            # if epoch == k_means_end_epoch:
            # update_label_embedding = False
            # print("##########Cease label embedding updates##########")

            # if epoch == WARM_UP_EPOCH:
            #     model.module.combiner.freeze_switch_modulation()
            # if epoch == 10:
            # model.module.combiner.freeze_all_modulation()

            train_epoch_log = train(
                cfg,
                model=model,
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
            inf_train_log = inference_train(model, train_dataloader, device, epoch, [1, 5, 10])
            wandb_run.log(inf_train_log)
            logger_epoch["inference_train"] = inf_train_log

        if cfg.control.train_2:  # KMeans update
            n_clusters = n_clusters_list[epoch]  # Number of clusters for the current epoch
            # An adaptive alpha which minimum 0.1 and maximum 0.9, slide depends on k_means_middle_epoch - k_means_start_epoch
            alpha = max(
                min(
                    (1 - (k_means_middle_epoch - epoch) / k_means_middle_epoch),
                    alpha_upper,
                ),
                0.01,
            )

            # Perform clustering and update embeddings by merging
            if k_means_start_epoch <= epoch < k_means_end_epoch and epoch % K_means_gap == 0:
                print(
                    f"##########Epoch {epoch}: Expected number of clusters: {n_clusters}##########"
                )

                # Load embeddings
                embedding_manager.load_embeddings()
                sample_ids, label_embedding = embedding_manager.get_all_embeddings()

                # Perform UMAP and clustering on unique embeddings
                print("##########Performing UMAP for computation##########")
                if label_embedding[0].shape[0] == 2:
                    umap_features_high = label_embedding.clone()
                    print("skip umap reduction")
                else:
                    umap_features_high = clustering.get_umap(
                        label_embedding, n_components=cfg.train_2.umap_components
                    )

                print("##########Performing Clustering##########")
                umap_labels, _ = clustering.get_hdbscan(
                    umap_features_high,
                    n_clusters=0,
                    method="eom",
                )
                # umap_labels, _ = clustering.get_kmeans(
                #     umap_features_high,
                #     n_clusters=50,
                # )
                unique_umap_labels = torch.unique(umap_labels)
                print(f"Unique UMAP labels: {unique_umap_labels.size(0)}")

                # Map clustering results back to the original embeddings #TODO: exclude hard update now
                # if epoch < k_means_middle_epoch:
                update_noise = "assign"  # TODO: We assign all noise points to the nearest cluster
                update_type = "soft"
                print("##########Performing soft clustering update##########")
                # else: #
                #     update_noise = "assign"
                #     update_type = "hard"
                #     high_lr = False
                #     print("##########Performing hard clustering update##########")

                updated_embeddings, cluster_centers, cluster_counts = clustering.hdbscan_update(
                    umap_labels=umap_labels,
                    original_embeddings=label_embedding,
                    update_type=update_type,
                    alpha=alpha,
                    update_noise=update_noise,
                    center_only=False,
                )

                # Find unique embeddings, and return the indices of the unique embeddings according to the descending order of the size of the cluster
                unique_embeddings, _ = torch.unique(updated_embeddings, return_inverse=True, dim=0)

                print(f"Unique embeddings after clustering update: {unique_embeddings.size(0)}")

                # Check how many embeddings have been updated by k-means
                differences = torch.any(label_embedding != updated_embeddings, dim=1)
                num_different_rows = torch.sum(differences).item()
                print(f"Number of rows updated by Clustering: {num_different_rows}")

                print(f"Number of true cluster centers after update: {cluster_centers.size(0)}")

                # if (epoch) % 5 == 0:
                # update the embeddings
                embedding_manager.update_all_chunks(
                    sample_ids, updated_embeddings
                )  # TODO: Stop update embeddings in the phase two training
                embedding_manager.load_embeddings()

                # Check if the saved embeddings are the same as the updated embeddings
                # updated_embeddings_2 = embedding_manager.get_all_embeddings()[1]

                # unique_embeddings_2, _ = torch.unique(updated_embeddings_2, return_inverse=True, dim=0)
                # print(f"Unique embeddings loaded: {unique_embeddings_2.size(0)}")

                # # # Check if embeddings have been updated
                # differences = torch.any(updated_embeddings != updated_embeddings_2, dim=1)
                # num_different_rows = torch.sum(differences).item()
                # assert num_different_rows == 0, f"Embeddings have not been updated after clustering, {num_different_rows} rows are different"

                # Find unique labels and their counts
                # Sort unique labels by descending count

                if cfg.control.save:
                    # Plot UMAP before clustering update

                    print("##########Performing UMAP for visualisation##########")
                    if label_embedding[0].shape[0] == 2:
                        umap_features, updated_umap_features = (
                            label_embedding.clone(),
                            updated_embeddings.clone(),
                        )
                        print("skip umap reduction for visualization")
                    else:
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

                center_sorted_indices = torch.argsort(cluster_counts, descending=True)
                unique_embeddings = cluster_centers[center_sorted_indices]
                print(f"Cluster Centers after sorting: {unique_embeddings.shape}")

                torch.save(
                    unique_embeddings,  # Increase the number of embeddings to save to 300
                    os.path.join(experiment_dir, "unique_embeddings.pt"),
                )

        if cfg.control.test:
            if unique_embeddings is not None:

                kmeans = KMeans(n_clusters=min(50, unique_embeddings.shape[0])).fit(
                    unique_embeddings.cpu().numpy()
                )
                centroids = kmeans.cluster_centers_
                # Find closest real embedding to each centroid

                indices = np.argmin(cdist(centroids, unique_embeddings), axis=1)
                representatives = unique_embeddings[indices]
                print("##########Testing test dataset##########")
                inf_test_log = inference_test(
                    model, processor, test_dataloader, representatives, epoch, device
                )
                logger_epoch["inference_test"] = inf_test_log
                wandb_run.log(inf_test_log)

            # elif epoch >= k_means_middle_epoch:
            #     # print("##########No clustering##########")
            #     # # Load embeddings
            #     # embedding_manager.load_embeddings()
            #     # sample_ids, label_embedding = embedding_manager.get_all_embeddings()
            #     # unique_embeddings, _ = torch.unique(label_embedding, return_inverse=True, dim=0)
            #     # print(f"Unique embeddings: {unique_embeddings.size(0)}")

            #     unique_embeddings = torch.load(
            #         os.path.join(experiment_dir, "unique_embeddings.pt")
            #     )

        if cfg.control.save_per_epoch is True and unique_embeddings is not None:
            # Save model, epoch, optimizer, scheduler
            folder_manager.save_model(model, checkpoint_dir, epoch)
            cluster_folder = folder_manager.get_cluster_folder(experiment_dir)
            torch.save(
                unique_embeddings,
                os.path.join(cluster_folder, f"unique_embeddings_{epoch}.pt"),
            )

        # Save logger per epoch
        logger.append(logger_epoch)
        OmegaConf.save(config=cfg, f=os.path.join(experiment_dir, "config.yaml"))
        folder_manager.save_final_model(model.module.combiner, experiment_dir)

    # Save final model and merge history
    # folder_manager.save_final_model(model, experiment_dir)
    # OmegaConf.save(config=cfg, f=os.path.join(experiment_dir, "config.yaml"))

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
    wandb.require("core")  # type: ignore
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
