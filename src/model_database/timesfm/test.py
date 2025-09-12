import os
import pickle
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model_database.timesfm.finetune import TimeSeriesDataset
from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder


"""
export PYTHONPATH=$(pwd)
poetry run streamlit run src/model_database/timesfm/test.py
"""
current_working_dir = os.getcwd()
REPO = "google/timesfm-2.0-500m-pytorch"
HORIZON = 12
CONTEXT = 24
NUM_LAYERS = {"google/timesfm-1.0-200m-pytorch": 20, "google/timesfm-2.0-500m-pytorch": 50}


# ---------------------------
# Cache model loading
# ---------------------------
@st.cache_resource
def get_model(from_dir: Optional[str], load_weights: bool = True):
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
            checkpoint_path = os.path.join(snapshot_download(repo_id), "torch_model.ckpt")
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(loaded_checkpoint)
    return model, hparams, tfm._model_config


# ---------------------------
# Load data once
# ---------------------------
@st.cache_resource
def load_data():
    root = f"{current_working_dir}/src/database"
    file_path = "TimeGAN/payroll_36"

    with open(os.path.join(root, file_path, "original_data.pkl"), "rb") as f:
        ori_data = pickle.load(f)
    ori_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in ori_data]

    with open(os.path.join(root, file_path, "generated_data.pkl"), "rb") as f:
        syn_data = pickle.load(f)
    syn_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in syn_data]

    test_data = [
        np.sin(np.linspace(0, 20, 100)),
        np.sin(np.linspace(0, 20, 200)),
        np.sin(np.linspace(0, 20, 400)),
    ]

    return {"ori": ori_data, "syn": syn_data, "test": test_data}


# ---------------------------
# Cache forecast per dataset key
# ---------------------------
@st.cache_data
def run_forecast(data_key: str, model_type: Literal["finetuned", "pretrained"]):
    if model_type == "finetuned":
        from_dir = f"{current_working_dir}/model_checkpoints/TimesFM"
    else:
        from_dir = None
    model, _, tfm_config = get_model(from_dir=from_dir, load_weights=True)
    model.eval()
    data_map = load_data()
    data = data_map[data_key]
    test_dataset = TimeSeriesDataset(
        data,
        context_length=CONTEXT,
        horizon_length=HORIZON,
        freq_type=1,
        actual_horizon=tfm_config.horizon_len,
        actual_context=tfm_config.patch_len,
    )
    test_dl = DataLoader(test_dataset, batch_size=32)
    forecasts, ground_truth = [], []
    with torch.no_grad():
        for i, (context, pad, freq, horizon) in enumerate(tqdm(test_dl, desc="Getting forecasts")):
            if i > 5:
                break
            predictions = model(context, pad.float(), freq)
            predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
            last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]
            horizon = horizon[:, :HORIZON]
            last_patch_pred = last_patch_pred[:, :HORIZON]
            forecasts = forecasts + list(last_patch_pred.unbind(0))
            # full_predictions = full_predictions = list(full_pred.unbind(0))
            ground_truth = ground_truth + list(horizon.unbind(0))
    return forecasts, ground_truth


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("TimesFM Forecast Viewer")

# Sidebar controls
data_key = st.sidebar.selectbox("Select dataset", ["ori", "syn", "test"])
model_type = st.sidebar.selectbox("Select model", ["finetuned", "pretrained"])
num_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=500, value=5)

# Run button
with st.spinner(f"Running forecast for `{data_key}`..."):
    # Load data and run forecasting
    data_map = load_data()
    data = data_map[data_key]
    point_forecasts, ground_truth = run_forecast(data_key, model_type=model_type)

    st.write(f"### Showing {num_samples} samples from `{data_key}` data")

    # Display forecasts for each sample
    for i in range(min(num_samples, len(data))):
        series = data[i]
        forecast = point_forecasts[i]

        # Convert tensors to numpy if needed
        if torch.is_tensor(series):
            series = series.cpu().numpy()
        if torch.is_tensor(forecast):
            forecast = forecast.cpu().numpy()

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        history_x = range(len(series))
        ax.plot(history_x, series, label="History", color="blue", linewidth=2)

        # Plot point forecast
        forecast_x = range(len(series), len(series) + len(forecast))
        ax.plot(forecast_x, forecast, label="Forecast", linestyle="--", color="red", linewidth=2)

        # Plot ground truth if available and requested
        if ground_truth is not None and i < len(ground_truth):
            gt = ground_truth[i]
            if torch.is_tensor(gt):
                gt = gt.cpu().numpy()
            ax.plot(forecast_x, gt, label="Ground Truth", color="green", linewidth=2, alpha=0.8)

        # Customize the plot
        ax.legend()
        ax.set_title(f"Sample {i+1} - {data_key} ({model_type})")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # Add vertical line to separate history from forecast
        ax.axvline(x=len(series) - 0.5, color="black", linestyle=":", alpha=0.5)

        st.pyplot(fig)
        plt.close(fig)  # Prevent memory leaks

        # Always display metrics if ground truth is available
        if ground_truth is not None and i < len(ground_truth):
            gt = ground_truth[i]
            if torch.is_tensor(gt):
                gt = gt.cpu().numpy()

            # Calculate metrics
            mse = np.mean((forecast - gt) ** 2)
            mae = np.mean(np.abs(forecast - gt))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("MSE", f"{mse:.4f}")
            with col2:
                st.metric("MAE", f"{mae:.4f}")

        st.divider()
