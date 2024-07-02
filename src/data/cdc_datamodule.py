import json
import os
from tqdm import tqdm

import torch
from PIL import Image, ImageFile

from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, config

from src.utils import EmbeddingManager, FolderManager

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated (corrupted) images

custom_download_path = "/data/SSD2/HF_datasets"
config.HF_DATASETS_CACHE = custom_download_path


class CDC_train(Dataset):

    def __init__(
        self, annotation_path, image_path, preprocess, embedding_manager, ratio=0.1
    ):
        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.vis_processors = preprocess
        self.embedding_manager = embedding_manager

        # Assign unique numeric IDs to each sample
        self.sample_ids = {i: idx for idx, i in enumerate(range(len(self.annotations)))}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.vis_processors(raw_image, return_tensors="pt")
        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) == str
            else self.annotations[idx]["caption"][0]
        )

        sample_id = self.sample_ids[idx]
        embedding = self.embedding_manager.get_embedding(sample_id)
        label_embedding = torch.tensor(embedding, dtype=torch.float32)

        return image_input, raw_text, label_embedding, sample_id


class CDC_test(Dataset):

    def __init__(
        self, annotation_path, image_path, preprocess, ratio=0.1
    ):
        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.vis_processors = preprocess

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.vis_processors(raw_image, return_tensors="pt")
        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) == str
            else self.annotations[idx]["caption"][:]
        )

        return image_input, raw_text


def test():
    from transformers import AutoProcessor, CLIPModel, AutoTokenizer, AutoImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Load annotations
    annotation_path_train = (
        "/data/SSD/flickr8k/annotations/train.json"  # 1 caption per image for flickr8k
    )
    image_path = "/data/SSD/flickr8k/Images"
    keyword = "clip-small"

    # Initialize FolderManager
    folder_manager = FolderManager(base_log_dir="/project/Deep-Clustering/res")

    # Initialize experiment
    experiment_dir = folder_manager.initialize_experiment(keyword)

    # Initialize embedding manager
    annotations = json.load(open(annotation_path_train))
    embedding_manager = EmbeddingManager(
        annotations, embedding_dim=512, chunk_size=10000, hdf5_dir=experiment_dir
    )

    # Initialize IMP dataset
    imp_train = CDC_train(
        annotation_path_train, image_path, preprocess, embedding_manager, ratio=1
    )

    imp_train_loader = DataLoader(imp_train, batch_size=16, shuffle=True)

    # Test IMP dataset
    for i, (image_input, raw_text, label_embedding, sample_id) in enumerate(
        tqdm(imp_train_loader)
    ):
        image_input = image_input.to(device)
        text_input = tokenizer(
            raw_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(device)
        label_embedding = label_embedding.to(device)
        image_features = model.vision_model(**image_input).last_hidden_state
        text_features = model.text_model(**text_input).last_hidden_state

        if i == 0:
            print(f"Image features: {image_features.shape}")
            print(f"Text features: {text_features.shape}")
            print(f"Label: {label_embedding.shape}")

        for ip, le in zip(sample_id, label_embedding):
            embedding_manager.update_embedding(ip, le.data)

    # annotation_path_val = "/data/SSD/flickr8k/annotations/val.json"
    # annotation_path_test = "/data/SSD/flickr8k/annotations/test.json"
    # imp_val = IMP_test(annotation_path_val, image_path, preprocess)
    # imp_test = IMP_test(annotation_path_test, image_path, preprocess)
    # imp_val_loader = DataLoader(imp_val, batch_size=32, shuffle=False)
    # for i, (image_input, raw_text) in enumerate(imp_val_loader):
    #     image_input = image_input.to(device)
    #     raw_text_list = []
    #     for b in range(32):
    #         for i in range(5):
    #             raw_text_list.append(raw_text[i][b])
    #     raw_text = raw_text_list
    #     text_input = tokenizer(
    #         raw_text, return_tensors="pt", padding="max_length", max_length=77
    #     ).to(device)
    #     image_features = model.vision_model(**image_input).last_hidden_state
    #     text_features = model.text_model(**text_input).last_hidden_state
    #     print(f"Image features: {image_features.shape}")
    #     print(f"Text features: {text_features.shape}")
    #     break


def main():
    test()


if __name__ == "__main__":
    main()
