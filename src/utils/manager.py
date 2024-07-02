import json
import os
from altair import sample
import h5py
import shutil
from datetime import datetime
from filelock import FileLock
from collections import defaultdict

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingManager:
    def __init__(
        self,
        annotations,
        embedding_dim=512,
        chunk_size=10000,
        hdf5_dir="embeddings",
        load_existing=False,
    ):
        self.annotations = annotations
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.hdf5_dir = hdf5_dir
        self.load_existing = load_existing

        # Create directory for HDF5 files if it does not exist
        os.makedirs(self.hdf5_dir, exist_ok=True)

        # Initialize HDF5 files and store embeddings in chunks
        self.chunk_files = []
        self.index_mapping = {}
        self.embedding_references = {}
        self.merge_history = defaultdict(set)  # Track merge history

        if load_existing:
            self.load_embeddings()
        else:
            self.initialize_embeddings()

    def initialize_embeddings(self):

        for chunk_idx in range(0, len(self.annotations), self.chunk_size):
            chunk_file = os.path.join(
                self.hdf5_dir, f"embeddings_{chunk_idx // self.chunk_size}.h5"
            )
            self.chunk_files.append(chunk_file)
            with h5py.File(chunk_file, "a") as hdf5_file:
                for i, item in enumerate(
                    self.annotations[chunk_idx : chunk_idx + self.chunk_size]
                ):
                    sample_id = i + chunk_idx  # Assign unique numeric ID

                    if str(sample_id) not in hdf5_file:
                        hdf5_file.create_dataset(
                            str(sample_id), data=torch.randn(self.embedding_dim).numpy()
                        )

                    self.index_mapping[sample_id] = (chunk_file, sample_id)
                    self.embedding_references[sample_id] = sample_id

    def load_embeddings(self):
        for file_name in os.listdir(self.hdf5_dir):
            if file_name.endswith(".h5"):
                chunk_file = os.path.join(self.hdf5_dir, file_name)
                self.chunk_files.append(chunk_file)
                with h5py.File(chunk_file, "r") as hdf5_file:
                    for sample_id in hdf5_file.keys():
                        sample_id_int = int(sample_id)
                        self.index_mapping[sample_id_int] = (chunk_file, sample_id_int)
                        self.embedding_references[sample_id_int] = sample_id_int

        # Log the loaded index mapping for debugging
        # print(f"Loaded index mapping: {self.index_mapping}")

    def get_embedding(self, sample_id):
        chunk_file, _ = self.index_mapping[sample_id]
        with h5py.File(chunk_file, "r") as hdf5_file:
            embedding = hdf5_file[str(self.embedding_references[sample_id])][:]
        return torch.tensor(embedding, dtype=torch.float32)

    # def update_embedding(self, sample_id, new_embedding):
    #     sample_id = int(sample_id)
    #     new_embedding = new_embedding.clone().detach()
    #     new_embedding = (
    #         new_embedding.cpu() if new_embedding.device != "cpu" else new_embedding
    #     )
    #     chunk_file, _ = self.index_mapping[sample_id]
    #     with h5py.File(chunk_file, "a") as hdf5_file:
    #         hdf5_file[str(self.embedding_references[sample_id])][
    #             :
    #         ] = new_embedding.numpy()

    def update_embedding(self, sample_id, new_embedding):
        sample_id = int(sample_id)
        new_embedding = new_embedding.clone().detach()
        new_embedding = (
            new_embedding.cpu() if new_embedding.device != "cpu" else new_embedding
        )
        chunk_file, _ = self.index_mapping[sample_id]
        with h5py.File(chunk_file, "a") as hdf5_file:
            hdf5_file[str(self.embedding_references[sample_id])][
                :
            ] = new_embedding.numpy()

    def share_embedding(self, sample_id1, sample_id2):
        self.embedding_references[sample_id2] = self.embedding_references[sample_id1]
        self.merge_history[sample_id1].add(sample_id2)
        self.merge_history[sample_id2].add(sample_id1)

    def save_embeddings_to_new_folder(self, new_hdf5_dir):
        # Create new directory for HDF5 files if it does not exist
        os.makedirs(new_hdf5_dir, exist_ok=True)

        # Copy old HDF5 files to the new folder
        for chunk_file in self.chunk_files:
            shutil.copy(chunk_file, new_hdf5_dir)

        # Update chunk_files to point to the new directory
        self.chunk_files = [
            os.path.join(new_hdf5_dir, os.path.basename(chunk_file))
            for chunk_file in self.chunk_files
        ]

    def get_proxy_embedding(self, sample_id):
        related_samples = self.merge_history[sample_id]
        all_samples = related_samples.union({sample_id})
        embeddings = [self.get_embedding(sid) for sid in all_samples]
        return torch.mean(torch.stack(embeddings), dim=0)

    def merge(self):
        # Placeholder function for merging embeddings
        pass


class FolderManager:
    def __init__(self, base_log_dir="logs"):
        self.base_log_dir = base_log_dir
        os.makedirs(self.base_log_dir, exist_ok=True)
        self.experiment_dir = None

    def initialize_experiment(self, keyword="empty_keyword"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(
            self.base_log_dir, f"{current_time}_{keyword}"
        )
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
        return checkpoint_dir, logs_dir # plot_dir

    def save_model(self, model, checkpoint_dir, epoch):
        model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)

    # def save_plot(self, fig, plot_dir, epoch):
    #     plot_path = os.path.join(plot_dir, f"umap_{epoch}.png")
    #     fig.savefig(plot_path)
    #     plt.close(fig)

    def save_metrics(self, metrics, logs_dir, epoch):
        metrics_path = os.path.join(logs_dir, f"metrics_epoch_{epoch}.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

    def save_final_model(self, model, experiment_dir):
        final_model_path = os.path.join(experiment_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)

    def save_merge_history(self, merge_history, experiment_dir):
        merge_history_path = os.path.join(experiment_dir, "merge_history.json")
        with open(merge_history_path, 'w') as f:
            json.dump(merge_history, f)

    # def save_config(self, config, experiment_dir):
    #     config_path = os.path.join(experiment_dir, "config.yaml")
    #     with open(config_path, 'w') as f:
    #         yaml.dump(config, f)


def find_device(model):
    return next(model.parameters()).device


def main():
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


if __name__ == "__main__":
    main()
