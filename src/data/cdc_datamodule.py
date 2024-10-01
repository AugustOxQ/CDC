import json
import os

import h5py
import torch
from datasets import config, load_dataset
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated (corrupted) images

custom_download_path = "/data/SSD2/HF_datasets"
config.HF_DATASETS_CACHE = custom_download_path


class FeatureExtractionDataset(Dataset):
    def __init__(self, annotation_path, image_path, processor, ratio=0.1):
        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.processor = processor

        # Assign unique numeric IDs to each sample
        self.sample_ids = {i: idx for idx, i in enumerate(range(len(self.annotations)))}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.processor(images=raw_image, return_tensors="pt")

        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) == str
            else self.annotations[idx]["caption"][0]
        )

        sample_id = self.sample_ids[idx]

        return image_input, raw_text, sample_id


class CDC_train_preextract(Dataset):
    def __init__(self, annotation_path, image_path, embedding_manager, feature_manager, ratio=0.1):
        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.embedding_manager = embedding_manager
        self.feature_manager = feature_manager
        self.chunk_size = feature_manager.chunk_size
        self.chunk_files = list(
            {
                self.feature_manager._get_chunk_file_and_idx(idx)[0]
                for idx in range(len(self.annotations))
            }
        )
        self.chunk_data = {}
        self.current_chunk = None

    def __len__(self):
        return len(self.chunk_files)

    def get_len(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        chunk_id = idx
        if self.current_chunk != chunk_id:
            self.current_chunk = chunk_id
            self.chunk_data = self.feature_manager.get_chunk(chunk_id)

        img_emb = self.chunk_data["img_features"][:]
        txt_emb = self.chunk_data["txt_features"][:]
        txt_full = self.chunk_data["txt_full"][:]
        sample_id = self.chunk_data["sample_ids"][:]

        img_emb = torch.tensor(img_emb, dtype=torch.float32)
        txt_emb = torch.tensor(txt_emb, dtype=torch.float32)
        txt_full = torch.tensor(txt_full, dtype=torch.float32)

        sample_id_2, embedding = self.embedding_manager.get_chunk_embeddings(chunk_id)

        # Turn sample_id into a list of integers
        sample_id = [int(i) for i in sample_id]
        sample_id_2 = [int(i) for i in sample_id_2]

        assert len(sample_id) == len(
            sample_id_2
        ), f"Sample ID length mismatch: expected {len(sample_id)}, got {len(sample_id_2)}"

        assert (
            sample_id[:10] == sample_id_2[:10]
        ), f"Sample ID mismatch: expected {sample_id[:10]}, got {sample_id_2[:10]}"

        label_embedding = torch.tensor(embedding, dtype=torch.float32)

        return img_emb, txt_emb, txt_full, label_embedding, sample_id


class CDC_test(Dataset):
    def __init__(self, annotation_path, image_path, processor, ratio=0.1):
        self.annotations = json.load(open(annotation_path))[:1000]
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.processor = processor
        self.captions_per_image = 1 if type(self.annotations[0]["caption"]) == str else 5
        print(f"Captions per image: {self.captions_per_image}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.processor(images=raw_image, return_tensors="pt")
        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) == str
            else self.annotations[idx]["caption"][:5]
        )

        return image_input, raw_text


def test():
    from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer, CLIPModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # preprocess = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def main():
    test()


if __name__ == "__main__":
    main()
