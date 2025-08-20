import streamlit as st
import pickle
import numpy as np
import os

"""
poetry run streamlit run src/dataset_preparation/synthetic/visualize.py
"""
root = "/mnt/c/Users/elzhang/Documents/payroll/model_checkpoints/timegan"
file_path = "stock/2025-08-20_12-58-35"
data_path = os.path.join(root, file_path, "original_data.pkl")
synthetic_path = os.path.join(root, file_path, "generated_data.pkl")

with open(data_path, "rb") as f:
    ori_data = pickle.load(f)

# Flatten last dimension if it's 1
flattened_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in ori_data]

num_samples = len(flattened_data)
seq_len = flattened_data[0].shape[0]

st.title("Original Data Visualization")

# Slider to select sample
sample_idx = st.slider("Select sample index", 0, num_samples-1, 0)

st.subheader(f"Sample {sample_idx} (length {seq_len})")
sample_data = flattened_data[sample_idx]

# Line chart
st.line_chart(sample_data)

# Optionally show raw values
if st.checkbox("Show raw data"):
    st.write(sample_data)