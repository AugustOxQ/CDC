import json
import os

import torch
from datasets import config
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To handle truncated (corrupted) images

custom_download_path = "/data/SSD2/HF_datasets"
config.HF_DATASETS_CACHE = custom_download_path


class FeatureExtractionDataset(Dataset):
    def __init__(self, annotation_path: str, image_path: str, processor, ratio=0.1) -> None:
        """
        Initialize the FeatureExtractionDataset class.

        Parameters
        ----------
        annotation_path : str
            Path to the annotation file, expected to be a JSON file.
        image_path : str
            Path to the directory containing the images.
        processor : object
            A processor object for processing the images.
        ratio : float, optional
            The ratio of samples to use from the annotation file, by default 0.1.

        Attributes
        ----------
        annotations : list
            A list of annotations loaded from the JSON file, reduced by the given ratio.
        image_path : str
            The path to the image directory.
        processor : object
            The processor object for processing the images.
        sample_ids : dict
            A dictionary assigning unique numeric IDs to each sample in the annotations list.
        """

        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.processor = processor

        # Assign unique numeric IDs to each sample
        self.sample_ids = {i: idx for idx, i in enumerate(range(len(self.annotations)))}

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve the processed image and textual annotation for a given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - image_input: The processed image tensor.
            - raw_text: The caption or text associated with the image.
            - sample_id: The unique numeric ID assigned to the sample.
        """

        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.processor(images=raw_image, return_tensors="pt")

        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) is str
            else self.annotations[idx]["caption"][0]
        )

        sample_id = self.sample_ids[idx]

        return image_input, raw_text, sample_id


class CDC_train_preextract(Dataset):
    def __init__(
        self,
        annotation_path: str,
        image_path: str,
        embedding_manager,
        feature_manager,
        ratio: float = 0.1,
    ) -> None:
        """
        Initialize the CDC_train_preextract class.

        Parameters
        ----------
        annotation_path : str
            Path to the annotation file, expected to be a JSON file.
        image_path : str
            Path to the directory containing the images.
        embedding_manager : object
            An object that manages the embedding process.
        feature_manager : object
            An object responsible for managing feature extraction.
        ratio : float, optional
            The ratio of samples to use from the annotation file, by default 0.1.

        Attributes
        ----------
        annotations : list
            A list of annotations loaded from the JSON file, reduced by the given ratio.
        image_path : str
            The path to the image directory.
        embedding_manager : object
            The embedding manager object.
        feature_manager : object
            The feature manager object.
        chunk_size : int
            The size of each chunk, as defined by the feature manager.
        chunk_files : list
            A list of chunk files used for feature extraction.
        chunk_data : dict
            A dictionary to store the current chunk's data.
        current_chunk : int or None
            The ID of the current chunk being processed.
        """

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
        """
        Get the number of chunks in the dataset.

        Returns
        -------
        int
            The number of chunks in the dataset.
        """
        return len(self.chunk_files)

    def get_len(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve the processed image, textual annotation, and label embedding for a given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - img_emb: The image embedding tensor, shape (1, embedding_dim).
            - txt_emb: The textual annotation embedding tensor, shape (1, embedding_dim).
            - txt_full: The full textual annotation tensor, shape (1, sequence_length).
            - label_embedding: The label embedding tensor, shape (1, embedding_dim).
            - sample_id: The unique numeric ID assigned to the sample.
        """

        chunk_id = idx
        if (
            self.current_chunk != chunk_id
        ):  # TODO: This might not be the most efficient way to handle this
            self.current_chunk = chunk_id
            self.chunk_data = self.feature_manager.get_chunk(chunk_id)

        img_emb = self.chunk_data["img_features"][:]
        # img_full = self.chunk_data["img_full"][:]
        txt_emb = self.chunk_data["txt_features"][:]
        txt_full = self.chunk_data["txt_full"][:]
        sample_id = self.chunk_data["sample_ids"][:]

        img_emb = torch.tensor(img_emb, dtype=torch.float32)
        # img_full = torch.tensor(img_full, dtype=torch.float32)
        txt_emb = torch.tensor(txt_emb, dtype=torch.float32)
        txt_full = torch.tensor(txt_full, dtype=torch.float32)

        sample_id_2, embedding = self.embedding_manager.get_chunk_embeddings(chunk_id)

        # Turn sample_id into a list of integers
        sample_id = [int(i) for i in sample_id]
        sample_id_2 = [int(i) for i in sample_id_2]
        if idx % 100 == 0:
            assert len(sample_id) == len(
                sample_id_2
            ), f"Sample ID length mismatch: expected {len(sample_id)}, got {len(sample_id_2)}"

            assert (
                sample_id[:10] == sample_id_2[:10]
            ), f"Sample ID mismatch: expected {sample_id[:10]}, got {sample_id_2[:10]}"

        label_embedding = torch.tensor(embedding, dtype=torch.float32)

        return img_emb, txt_emb, txt_full, label_embedding, sample_id


class CDC_test(Dataset):
    def __init__(
        self,
        annotation_path: str,
        image_path: str,
        processor,
        ratio: float = 0.1,
        crop_num: int = 5,
    ):
        """Initialize the CDC_test class.

        Parameters
        ----------
        annotation_path : str
            Path to the annotation file
        image_path : str
            Path to the image directory
        processor : object
            A processor object for processing the images
        ratio : float, optional
            The ratio of samples to use from the annotation file, by default 0.1

        Attributes
        ----------
        annotations : list
            The list of annotations
        image_path : str
            The path to the image directory
        processor : object
            The processor object for processing the images
        captions_per_image : int
            The number of captions per image, either 1 or 5
        """
        self.annotations = json.load(open(annotation_path))
        self.annotations = self.annotations[: int(len(self.annotations) * ratio)]
        self.image_path = image_path
        self.processor = processor
        self.crop_num = crop_num
        self.caption_length_list = []
        for i in range(len(self.annotations)):
            if type(self.annotations[i]["caption"]) is str:
                self.caption_length_list.append(1)
            else:
                self.caption_length_list.append(len(self.annotations[i]["caption"]))
        self.captions_per_image = (
            1
            if type(self.annotations[0]["caption"]) is str
            else len(self.annotations[0]["caption"])
        )
        self.captions_per_image = min(self.captions_per_image, self.crop_num)
        print(f"Captions per image: {self.captions_per_image}")

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieve the processed image and up to 5 textual annotations for a given index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - image_input: The processed image tensor.
            - raw_text: The first 5 captions or text associated with the image, or a single caption if only one exists.
        """

        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_path, annotation["image"])
        raw_image = Image.open(img_path).convert("RGB")
        image_input = self.processor(images=raw_image, return_tensors="pt")
        if "pixel_values" in image_input:
            image_input["pixel_values"] = image_input["pixel_values"].squeeze()

        raw_text = (
            self.annotations[idx]["caption"]
            if type(self.annotations[idx]["caption"]) is str
            else self.annotations[idx]["caption"][: self.crop_num]
        )

        return image_input, raw_text


def testdataset(): ...


def main():
    testdataset()


if __name__ == "__main__":
    main()
