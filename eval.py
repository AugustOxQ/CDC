import json
import os
import warnings
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

import hydra
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
from src.metric.loss import CosineLoss, MeanSquareLoss
from src.models.cdc import CDC
from src.models.components.clustering import Clustering
from src.utils import EmbeddingManager, FolderManager, evalrank, calculate_n_clusters, plot_umap, FeatureManager

from train import cosine_similarity, compute_recall_at_k, inference_test

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_checkpoint(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def run(cfg):
    model_path = "/project/Deep-Clustering/res/20240716_180924_flickr30k-preextracted/checkpoints/model_epoch_25.pth"
    label_embedding_path = "/project/Deep-Clustering/res/20240716_180924_flickr30k-preextracted/epoch_25_kmupdate"
    
    print("Loading model")
    model = CDC()
    model = load_model_checkpoint(model, model_path).to(device)
    
    preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Initialize feature manager
    feature_manager = FeatureManager(cfg.dataset.extract_path, chunk_size=cfg.train.batch_size)
    
    annotations = json.load(open(cfg.dataset.train_path))
    annotations = annotations[: int(len(annotations) * cfg.dataset.ratio)]
    embedding_manager = EmbeddingManager(annotations, embedding_dim=512, chunk_size=10000, hdf5_dir=label_embedding_path)
    
    # Create Train and Test dataloader
    train_dataset = CDC_train(
        annotation_path=cfg.dataset.train_path,
        image_path=cfg.dataset.img_path,
        embedding_manager=embedding_manager,
        feature_manager=feature_manager,
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
    
    print("Loading embeddings")
    # Determine the number of clusters
    label_embedding = torch.stack(
            [
                embedding_manager.get_embedding(sample_id)
                for sample_id in range(len(train_dataset))
            ]
        )
    
    # Identify unique embeddings
    unique_embeddings, _ = torch.unique(
            label_embedding, return_inverse=True, dim=0
        )
    
    print("Calculating number of clusters")
    print("Number of unique embeddings: ", len(unique_embeddings))

@hydra.main(config_path="configs", config_name="flickr30k", version_base=None)
def main(cfg):
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run(cfg)
    
if __name__ == "__main__":
    main()