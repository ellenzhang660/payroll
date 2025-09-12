import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import timesfm


"""
poetry run streamlit run src/model_database/timesfm/test.py
"""
current_working_dir = os.getcwd()


# ---------------------------
# Cache model loading
# ---------------------------
@st.cache_resource
def load_tfm():
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=12,
            num_layers=50,
            use_positional_embedding=False,
            context_len=32,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )


# ---------------------------
# Load data once
# ---------------------------
@st.cache_resource
def load_data():
    root = f"{current_working_dir}/src/database"
    file_path = "TimeGAN/payroll"

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
def run_forecast(data_key: str):
    tfm = load_tfm()
    data_map = load_data()
    data = data_map[data_key]
    freq = [1] * len(data)
    point_forecast, _ = tfm.forecast(data, freq=freq)
    return point_forecast


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("TimesFM Forecast Viewer")

# Sidebar controls
data_key = st.sidebar.selectbox("Select dataset", ["ori", "syn", "test"])
num_samples = st.sidebar.number_input("Number of samples", min_value=1, max_value=500, value=5)

# Run button
if st.button("Run Forecast"):
    with st.spinner(f"Running forecast for `{data_key}`…"):
        data_map = load_data()
        data = data_map[data_key]
        point_forecast = run_forecast(data_key)

        st.write(f"### Showing {num_samples} samples from `{data_key}` data")

        for i in range(min(num_samples, len(data))):
            series = data[i]
            forecast = point_forecast[i]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(len(series)), series, label="History")
            ax.plot(
                range(len(series), len(series) + len(forecast)),
                forecast,
                label="Forecast",
                linestyle="--",
            )
            ax.legend()
            ax.set_title(f"Sample {i}")
            st.pyplot(fig)
