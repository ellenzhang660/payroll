import logging
import math
import os
import pickle
from os import path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder


DATA_ROOT = "/mnt/c/Users/elzhang/Documents/payroll/src/database"
FILE_PATH = "TimeGAN/payroll_36"
CONTEXT_LENGTH = 24  # will be padded to multiple of 32
HORIZON_LENGTH = 12  # will be padded to 128
REPO = "google/timesfm-2.0-500m-pytorch"
NUM_LAYERS = {"google/timesfm-1.0-200m-pytorch": 20, "google/timesfm-2.0-500m-pytorch": 50}
FREQ = 1  # 1 for monthly, 0 for

"""

export PYTHONPATH=$(pwd)
poetry run python src/model_database/timesfm/finetune.py
Finetunes on syntehtic data from TimeGAN
Code from https://github.com/google-research/timesfm/blob/master/notebooks/finetuning_torch.ipynb
"""

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"  # or INFO, WARNING, ERROR
)


class TimeSeriesDataset(Dataset):
    """Dataset for time series data compatible with TimesFM."""

    def __init__(
        self,
        series: np.ndarray,
        context_length: int,
        horizon_length: int,
        actual_horizon: int,
        actual_context: int,
        freq_type: int = 0,
    ):
        """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
        if freq_type not in [0, 1, 2]:
            raise ValueError("freq_type must be 0, 1, or 2")

        self.series = series
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.padded_context_length = math.ceil(context_length / actual_context) * actual_context
        self.padded_horizon_length = actual_horizon
        self.freq_type = freq_type
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        """Prepare sliding window samples from the time series."""
        self.samples = []

        for idx in range(len(self.series)):
            self.series[idx] = np.concatenate(
                [np.zeros(max(0, self.context_length + self.horizon_length - len(self.series[idx]))), self.series[idx]]
            )
            for start_idx in range(0, len(self.series[idx]) - self.context_length - self.horizon_length + 1):
                end_idx = start_idx + self.context_length
                x_context = self.series[idx][start_idx:end_idx]
                x_future = self.series[idx][end_idx : end_idx + self.horizon_length]

                # Pad to fixed length if necessary, shouldn't happen though
                if len(x_context) < self.padded_context_length:
                    padding_length = self.padded_context_length - len(x_context)
                    x_context = np.concatenate([np.zeros(padding_length), x_context])

                if len(x_future) < self.padded_horizon_length:
                    padding_length = self.padded_horizon_length - len(x_future)
                    x_future = np.concatenate([x_future, np.zeros(padding_length)])
                self.samples.append((x_context, x_future))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_context, x_future = self.samples[index]

        x_context = torch.tensor(x_context, dtype=torch.float32)
        x_future = torch.tensor(x_future, dtype=torch.float32)

        # input_padding = torch.zeros_like(x_context)
        input_padding = torch.cat(
            [torch.ones(self.padded_context_length - self.context_length), torch.zeros(self.context_length)]
        )
        freq = torch.tensor([self.freq_type], dtype=torch.long)

        return x_context, input_padding, freq, x_future


def prepare_datasets(
    series: np.ndarray,
    context_length: int,
    horizon_length: int,
    actual_context: int,
    actual_horizon: int,
    freq_type: int = 0,
    train_split: float = 0.8,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_size = int(len(series) * train_split)
    train_data = series[:train_size]
    val_data = series[train_size:]

    # Create datasets with specified frequency type
    train_dataset = TimeSeriesDataset(
        train_data,
        context_length=context_length,
        horizon_length=horizon_length,
        freq_type=freq_type,
        actual_context=actual_context,
        actual_horizon=actual_horizon,
    )

    if train_split != 1:
        val_dataset = TimeSeriesDataset(
            val_data,
            context_length=context_length,
            horizon_length=horizon_length,
            freq_type=freq_type,
            actual_context=actual_context,
            actual_horizon=actual_horizon,
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset


def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
    """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    model.eval()

    x_context, x_padding, freq, x_future = val_dataset[0]
    x_context = x_context.unsqueeze(0)  # Add batch dimension
    x_padding = x_padding.unsqueeze(0)
    freq = freq.unsqueeze(0)
    x_future = x_future.unsqueeze(0)

    device = next(model.parameters()).device
    x_context = x_context.to(device)
    x_padding = x_padding.to(device)
    freq = freq.to(device)
    x_future = x_future.to(device)

    with torch.no_grad():
        predictions = model(x_context, x_padding.float(), freq)
        predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
        last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

    context_vals = x_context[0].cpu().numpy()
    future_vals = x_future[0].cpu().numpy()
    pred_vals = last_patch_pred[0].cpu().numpy()
    context_len = len(context_vals)
    horizon_len = len(future_vals)

    plt.figure(figsize=(12, 6))

    plt.plot(range(context_len), context_vals, label="Historical Data", color="blue", linewidth=2)

    plt.plot(
        range(context_len, context_len + horizon_len),
        future_vals,
        label="Ground Truth",
        color="green",
        linestyle="--",
        linewidth=2,
    )

    plt.plot(range(context_len, context_len + horizon_len), pred_vals, label="Prediction", color="red", linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("TimesFM Predictions vs Ground Truth")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.close()


def get_model(from_dir: Optional[str], load_weights: bool = False):
    """
    from_dir: already finetuned model path
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id = REPO
    hparams = TimesFmHparams(
        backend=device,
        per_core_batch_size=32,
        horizon_len=128,
        num_layers=NUM_LAYERS[REPO],  # 20 for v1, 50 for v2
        use_positional_embedding=True,  # set to true for v1.0
        context_len=32,  # Context length can be anything up to 2048 in multiples of 32
    )
    tfm = TimesFm(hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

    model = PatchedTimeSeriesDecoder(tfm._model_config)
    if load_weights:
        if from_dir:
            checkpoint_path = f"{from_dir}/checkpoint.pth"
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(loaded_checkpoint["model_state_dict"])
        else:
            checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(loaded_checkpoint)
    return model, hparams, tfm._model_config


def get_data(
    context_len: int,
    horizon_len: int,
    actual_context: int,
    actual_horizon: int,
    freq_type: int = 0,
    type: Literal["finetune", "test"] = "finetune",
) -> Tuple[Dataset, Dataset]:

    if type == "test":
        path = os.path.join(DATA_ROOT, FILE_PATH, "original_data.pkl")
        split = 1.0
    else:
        path = os.path.join(DATA_ROOT, FILE_PATH, "generated_data.pkl")
        split = 0.8
    with open(path, "rb") as f:
        syn_data = pickle.load(f)
        time_series = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in syn_data]

    train_dataset, val_dataset = prepare_datasets(
        series=time_series,
        context_length=context_len,
        horizon_length=horizon_len,
        freq_type=freq_type,
        train_split=split,
        actual_context=actual_context,
        actual_horizon=actual_horizon,
    )

    print("Created datasets:")
    print(f"- {type} samples: {len(train_dataset)}")
    if val_dataset:
        print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Using frequency type: {freq_type}")
    return train_dataset, val_dataset


def single_cpu_finetune():
    """Basic example of finetuning TimesFM on stock data."""
    model, hparams, tfm_config = get_model(load_weights=True, from_dir=None)
    config = FinetuningConfig(
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-5,
        use_wandb=False,
        freq_type=FREQ,
        log_every_n_steps=10,
        val_check_interval=0.5,
        use_quantile_loss=True,
    )
    train_dataset, val_dataset = get_data(
        context_len=CONTEXT_LENGTH,
        horizon_len=HORIZON_LENGTH,
        freq_type=config.freq_type,
        actual_horizon=tfm_config.horizon_len,
        actual_context=tfm_config.patch_len,
    )
    finetuner = TimesFMFinetuner(
        model, config, save_dir=f"model_checkpoints/TimesFM_{REPO}", logger=logging.getLogger(__name__)
    )

    print("\nStarting finetuning...")
    results = finetuner.finetune(train_dataset=train_dataset, val_dataset=val_dataset)

    print("\nFinetuning completed!")
    print(f"Training history: {len(results['history']['train_loss'])} epochs")

    plot_predictions(
        model=model,
        val_dataset=val_dataset,
        save_path="timesfm_predictions.png",
    )


def evaluate():
    model, hparams, tfm_config = get_model(load_weights=True, from_dir="model_checkpoints/TimesFM")
    test_dataset, _ = get_data(
        context_len=CONTEXT_LENGTH,
        horizon_len=HORIZON_LENGTH,
        freq_type=FREQ,
        actual_horizon=tfm_config.horizon_len,
        actual_context=tfm_config.patch_len,
        type="test",
    )
    print("Evaluating")
    plot_predictions(
        model=model,
        val_dataset=test_dataset,
        save_path="timesfm_predictions_original.png",
    )


if __name__ == "__main__":
    single_cpu_finetune()
#   evaluate()
