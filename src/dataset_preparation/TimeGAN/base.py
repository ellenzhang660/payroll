import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Union

from dataset_preparation.TimeGAN.trainer.data_convert import dataset_to_numpy
from dataset_preparation.TimeGAN.trainer.timegan import timegan
from dataset_preparation.TimeGAN.trainer.visualization_metrics import visualization
from src.dataset.base import TimeSeriesData
from src.dataset_preparation.base import GenerateFakeData


@dataclass
class TimeGANParameters:
    """Configuration parameters for TimeGAN model training."""

    module: Literal["gru", "lstm", "lstmLN"] = "gru"
    hidden_dim: int = 24
    num_layer: int = 3
    iterations: int = 10000
    batch_size: int = 128

    def to_dict(self) -> dict[str, Union[str, int]]:
        """Convert dataclass to dictionary for compatibility with existing code."""
        return {
            "module": self.module,
            "hidden_dim": self.hidden_dim,
            "num_layer": self.num_layer,
            "iterations": self.iterations,
            "batch_size": self.batch_size,
        }


class TimeGANGenerator(GenerateFakeData):
    """
    TimeGAN model approach for generating synthetic data
    Loads in a trained model for generation
    To train one a TimeGAN model on new data, go to timegan.py in this folder
    """

    def __init__(self, dataset: TimeSeriesData, seq_len: int, parameters: Optional[TimeGANParameters]):
        super().__init__(dataset=dataset)
        self.seq_len = seq_len
        self.parameters = parameters or TimeGANParameters()

    def generate_fake_data(self):
        # load dataset
        ori_data = dataset_to_numpy(self.base, self.seq_len)

        # Create a unique checkpoint directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.checkpoint_dir = f"model_checkpoints/timegan/{repr(self.base)}/{timestamp}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # train
        generated_data = timegan(ori_data, self.parameters, self.checkpoint_dir)
        print("Finish Synthetic Data Generation")

        # save data
        with open(f"{self.checkpoint_dir}/generated_data.pkl", "wb") as f:
            pickle.dump(generated_data, f)

        # visualize
        self.test(ori_data=ori_data, generated_data=generated_data)

        return generated_data

    def test(self, ori_data, generated_data):
        fig = visualization(ori_data, generated_data, "pca")
        fig.savefig(f"{self.checkpoint_dir}/test_pca.png", dpi=300)
        fig = visualization(ori_data, generated_data, "tsne")
        fig.savefig(f"{self.checkpoint_dir}/test_tsne.png", dpi=300)

    def __str__(self) -> str:
        """Pretty-print all observer information about the dataset."""
        info_lines = [
            "TimeGAN Generator Configuration:",
            f"  Base Dataset: {self.base}",
            f"  Sequence Length: {self.seq_len}",
            "  Model Parameters:",
            f"    - Module: {self.parameters.module.upper()}",
            f"    - Hidden Dimensions: {self.parameters.hidden_dim}",
            f"    - Number of Layers: {self.parameters.num_layer}",
            f"    - Training Iterations: {self.parameters.iterations:,}",
            f"    - Batch Size: {self.parameters.batch_size}",
        ]

        # Add checkpoint directory info if it exists
        if hasattr(self, "checkpoint_dir") and self.checkpoint_dir:
            info_lines.append(f"  Checkpoint Directory: {self.checkpoint_dir}")

        return "\n".join(info_lines)

    def __repr__(self) -> str:
        return f"TimeGANGenerator(dataset={self.base!r}, seq_len={self.seq_len})"
