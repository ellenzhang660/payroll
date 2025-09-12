import os
import pickle

import streamlit as st


"""
poetry run streamlit run src/dataset_preparation/TimeGAN/visualize.py
"""

root = "/mnt/c/Users/elzhang/Documents/payroll/model_checkpoints/timegan"
file_path = "payroll"
data_path = os.path.join(root, file_path, "original_data.pkl")
synthetic_path = os.path.join(root, file_path, "generated_data.pkl")

# Load original data
with open(data_path, "rb") as f:
    ori_data = pickle.load(f)
ori_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in ori_data]

# Load synthetic data
with open(synthetic_path, "rb") as f:
    syn_data = pickle.load(f)
syn_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in syn_data]

# Assume both have same length
num_samples = min(len(ori_data), len(syn_data))
seq_len = ori_data[0].shape[0]

st.title("Original vs Synthetic Data")

# Single slider
idx = st.slider("Select sample index", 0, num_samples - 1, 0)

# Two columns for side-by-side comparison
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Original Sample {idx}")
    st.line_chart(ori_data[idx])

with col2:
    st.subheader(f"Synthetic Sample {idx}")
    st.line_chart(syn_data[idx])

# Optional raw values toggle
if st.checkbox("Show raw values"):
    st.write("Original:", ori_data[idx])
    st.write("Synthetic:", syn_data[idx])
