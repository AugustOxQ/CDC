import json
import os
import shutil
from collections import defaultdict
from datetime import datetime

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureManager:
    def __init__(self, features_dir, chunk_size):
        self.features_dir = features_dir
        # Create directory for feature files if it does not exist
        os.makedirs(self.features_dir, exist_ok=True)
        self.chunk_size = chunk_size
        self.index_mapping = {}
        self.feature_references = []

    def add_features_chunk(self, chunk_id, img_features, txt_features, txt_full, sample_ids):
        if len(img_features) != len(txt_features) or len(img_features) != len(sample_ids):
            raise ValueError(
                "Length of image features, text features, and sample IDs must be the same."
            )

        chunk_file = os.path.join(self.features_dir, f"chunk_{chunk_id}.pt")
        features = {
            "img_features": img_features,
            "txt_features": txt_features,
            "txt_full": txt_full,
            "sample_ids": sample_ids,
        }

        torch.save(features, chunk_file)

    def load_features(self):
        for file_name in os.listdir(self.features_dir):
            if file_name.endswith(".pt"):
                chunk_file = os.path.join(self.features_dir, file_name)

    def get_chunk(self, chunk_id):
        chunk_file = os.path.join(self.features_dir, f"chunk_{chunk_id}.pt")
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file {chunk_file} not found.")
        return torch.load(chunk_file)

    def _get_chunk_file_and_idx(self, sample_id):
        chunk_idx = sample_id % self.chunk_size
        chunk_file = os.path.join(self.features_dir, f"chunk_{sample_id // self.chunk_size}.pt")
        return chunk_file, chunk_idx

    def debug_print(self):
        print(f"Index Mapping: {self.index_mapping}")


class EmbeddingManager:
    def __init__(
        self,
        annotations,
        embedding_dim=512,
        chunk_size=1024,
        embeddings_dir="embeddings",
        load_existing=False,
        sample_ids_list: list[int] = [0, 1],
    ):
        self.annotations = annotations
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.embeddings_dir = embeddings_dir
        self.load_existing = load_existing

        self.sample_ids_list = sample_ids_list

        # Create directory for embedding files if it does not exist
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Initialize embedding files and store embeddings in chunks
        self.chunk_files = []
        self.index_mapping = {}
        self.embedding_references = {}
        self.merge_history = defaultdict(set)  # Track merge history

        if load_existing:
            self.load_embeddings()
        else:
            self.initialize_embeddings()

    def initialize_embeddings(self):
        mean = 0.0
        std_dev = 1.0
        # Initialize embeddings and store them in chunks
        for chunk_idx in range(0, len(self.sample_ids_list), self.chunk_size):
            chunk_file = os.path.join(
                self.embeddings_dir, f"embeddings_{chunk_idx // self.chunk_size}.pt"
            )
            self.chunk_files.append(chunk_file)
            embeddings = {}
            for _, sample_id in enumerate(
                self.sample_ids_list[chunk_idx : chunk_idx + self.chunk_size]
            ):
                # embeddings[sample_id] = torch.randn(
                #     self.embedding_dim
                # )  # normally distributed random embeddings
                # embeddings[sample_id] = torch.exp(torch.randn(self.embedding_dim) * std_dev + mean) # exponentially distributed random embeddings
                embeddings[sample_id] = torch.zeros(self.embedding_dim)  # zero embeddings
                self.index_mapping[sample_id] = (chunk_file, sample_id)
                self.embedding_references[sample_id] = sample_id

            torch.save(embeddings, chunk_file)

    def load_embeddings(self):
        # Load existing embeddings from chunk files
        self.chunk_files = []
        for file_name in sorted(os.listdir(self.embeddings_dir)):
            if file_name.endswith(".pt"):
                chunk_file = os.path.join(self.embeddings_dir, file_name)
                self.chunk_files.append(chunk_file)
                embeddings = torch.load(chunk_file)
                for sample_id in embeddings.keys():
                    self.index_mapping[sample_id] = (chunk_file, sample_id)
                    self.embedding_references[sample_id] = sample_id

    def get_chunk_embeddings(self, chunk_id):
        # Return embeddings from a specific chunk
        chunk_file = os.path.join(self.embeddings_dir, f"embeddings_{chunk_id}.pt")
        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file {chunk_file} not found.")
        all_embeddings = torch.load(chunk_file)

        sample_ids = all_embeddings.keys()
        label_embeddings = [all_embeddings[sample_id] for sample_id in sample_ids]

        return sample_ids, torch.stack(label_embeddings)

    def update_chunk_embeddings(self, chunk_id, sample_ids, new_embeddings):
        # Update embeddings in a specific chunk
        chunk_file = os.path.join(self.embeddings_dir, f"embeddings_{chunk_id}.pt")

        if not os.path.exists(chunk_file):
            raise FileNotFoundError(f"Chunk file {chunk_file} not found.")

        embeddings = torch.load(chunk_file)
        sample_ids = [int(sample_id) for sample_id in sample_ids]

        # check sample_ids are in embeddings.keys()
        assert all(sample_id in embeddings.keys() for sample_id in sample_ids)

        for sample_id, new_embedding in zip(sample_ids, new_embeddings):
            new_embedding = new_embedding.clone().detach().cpu()
            embeddings[sample_id] = new_embedding

        torch.save(embeddings, chunk_file)

    def get_all_embeddings(self):
        # Return unsorted embeddings by loading all chunks and concatenating them
        sample_ids_list = []
        label_embeddings_list = []
        for chunk_id in range(len(self.chunk_files)):
            sample_ids, label_embeddings = self.get_chunk_embeddings(chunk_id)
            sample_ids_list.extend(sample_ids)
            label_embeddings_list.append(label_embeddings)

        label_embeddings_list = torch.cat(label_embeddings_list)

        return sample_ids_list, label_embeddings_list

    def update_all_chunks(self, sample_ids, new_embeddings):
        chunk_size = self.chunk_size
        for i in range(0, len(sample_ids), chunk_size):
            self.update_chunk_embeddings(
                i // chunk_size,
                sample_ids[i : i + chunk_size],
                new_embeddings[i : i + chunk_size],
            )

    def save_embeddings_to_new_folder(self, new_embeddings_dir):
        # Create new directory for embedding files if it does not exist
        os.makedirs(new_embeddings_dir, exist_ok=True)

        # Copy old embedding files to the new folder
        for chunk_file in self.chunk_files:
            shutil.copy(chunk_file, new_embeddings_dir)

        # Update chunk_files to point to the new directory
        self.chunk_files = [
            os.path.join(new_embeddings_dir, os.path.basename(chunk_file))
            for chunk_file in self.chunk_files
        ]


class FolderManager:
    def __init__(self, base_log_dir="logs"):
        self.base_log_dir = base_log_dir
        os.makedirs(self.base_log_dir, exist_ok=True)
        self.experiment_dir = None

    def initialize_experiment(self, keyword="empty_keyword"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(self.base_log_dir, f"{current_time}_{keyword}")
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Create 'init' folder for initial embeddings
        init_dir = os.path.join(self.experiment_dir, "init")
        os.makedirs(init_dir, exist_ok=True)

        # Create 'plots' folder for storing plots
        plot_dir = os.path.join(self.experiment_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        return self.experiment_dir, init_dir, plot_dir

    def load_experiment(self, experiment_dir):
        self.experiment_dir = experiment_dir

        init_dir = os.path.join(self.experiment_dir, "init")
        plot_dir = os.path.join(self.experiment_dir, "plots")

        return init_dir, plot_dir

    def load_epoch_folder(self, epoch):
        if self.experiment_dir is None:
            raise ValueError("Experiment directory is not initialized.")

        epoch_dir = os.path.join(self.experiment_dir, f"epoch_{epoch}")
        if not os.path.exists(epoch_dir):
            raise ValueError(f"Epoch directory {epoch_dir} does not exist.")

        return epoch_dir

    def create_epoch_folder(self, epoch):
        if self.experiment_dir is None:
            raise ValueError("Experiment directory is not initialized.")
        epoch_dir = os.path.join(self.experiment_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        return epoch_dir

    def create_plot_folder(self, experiment_dir):
        plot_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        return plot_dir

    def create_checkpoint_folder(self, experiment_dir):
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return checkpoint_dir

    def create_logs_folder(self, experiment_dir):
        logs_dir = os.path.join(experiment_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir

    def create_directories(self, experiment_dir):
        # plot_dir = self.create_plot_folder(experiment_dir)
        checkpoint_dir = self.create_checkpoint_folder(experiment_dir)
        logs_dir = self.create_logs_folder(experiment_dir)
        return checkpoint_dir, logs_dir  # plot_dir

    def save_model(self, model, checkpoint_dir, epoch):
        model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)

    def save_metrics(self, metrics, logs_dir, epoch):
        metrics_path = os.path.join(logs_dir, f"metrics_epoch_{epoch}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def save_final_model(self, model, experiment_dir):
        final_model_path = os.path.join(experiment_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)

    def save_merge_history(self, merge_history, experiment_dir):
        merge_history_path = os.path.join(experiment_dir, "merge_history.json")
        with open(merge_history_path, "w") as f:
            json.dump(merge_history, f)


def find_device(model):
    return next(model.parameters()).device


def test_embedding_manager():
    import random

    # Set seed to be 42
    random.seed(42)
    # Create fake annotations
    annotations = list(range(5000))
    sample_ids_list = list(range(5000))
    # shuffle sample_ids
    sample_ids_list = random.sample(sample_ids_list, len(sample_ids_list))
    print(sample_ids_list[:15])
    embedding_manager = EmbeddingManager(
        annotations,
        embedding_dim=512,
        chunk_size=1000,
        embeddings_dir="/project/Deep-Clustering/res/test_embeddings",
        sample_ids_list=sample_ids_list,
    )
    embedding_manager.initialize_embeddings()
    embedding_manager.load_embeddings()
    sample_ids, embeddings = embedding_manager.get_all_embeddings()

    print(embeddings.shape)
    print(sample_ids[:15])

    new_embeddings = embeddings.clone()
    new_embeddings[:500] = torch.randn(500, 512)
    embedding_manager.update_all_chunks(sample_ids, new_embeddings)

    embedding_manager.load_embeddings()

    updated_sample_ids, updated_embeddings = embedding_manager.get_all_embeddings()
    print(updated_embeddings.shape)

    print(updated_sample_ids[:15])

    differences = torch.any(embeddings != updated_embeddings, dim=1)

    # Count number of unchanged rows
    num_different_rows = torch.sum(differences).item()
    print(f"Number of different rows: {num_different_rows}")
    print(f"Number of unchanged rows: {len(embeddings) - num_different_rows}")
    print(
        "This is the comparison between the original embeddings and the embeddings updated by update_all_chunks, should be 500 different rows"
    )

    new_embeddings_2 = embeddings[:100].clone()
    embedding_manager.update_chunk_embeddings(0, sample_ids[:100], new_embeddings_2)
    embedding_manager.load_embeddings()

    updated_sample_ids_2, updated_embeddings_2 = embedding_manager.get_all_embeddings()

    print(updated_sample_ids_2[:15])

    differences_2 = torch.any(updated_embeddings != updated_embeddings_2, dim=1)
    num_different_rows_2 = torch.sum(differences_2).item()
    print(f"Number of different rows: {num_different_rows_2}")
    print(f"Number of unchanged rows: {len(updated_embeddings) - num_different_rows_2}")
    print(
        "This is the comparison between the original embeddings and the embeddings updated by update_chunk_embeddings, should be 100 different rows"
    )

    differences_3 = torch.any(updated_embeddings_2 != embeddings, dim=1)
    num_different_rows_3 = torch.sum(differences_3).item()
    print(f"Number of different rows: {num_different_rows_3}")
    print(f"Number of unchanged rows: {len(embeddings) - num_different_rows_3}")
    print(
        "This is the comparison between the update_chunk_embeddings and update_all_chunks, should be 400 different rows"
    )


def main():
    test_embedding_manager()


if __name__ == "__main__":
    main()
