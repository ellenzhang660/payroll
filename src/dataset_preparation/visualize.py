import streamlit as st
import pickle
import numpy as np

file_path = "/mnt/c/Users/elzhang/Documents/payroll/src/database/synthetic_data/payroll/generated_data.pkl"

with open(file_path, "rb") as f:
    generated_data = pickle.load(f)

# Flatten last dimension if it's 1
flattened_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in generated_data]

num_samples = len(flattened_data)
seq_len = flattened_data[0].shape[0]

st.title("Synthetic Payroll Data Visualization")

# Slider to select sample
sample_idx = st.slider("Select sample index", 0, num_samples-1, 0)

st.subheader(f"Sample {sample_idx} (length {seq_len})")
sample_data = flattened_data[sample_idx]

# Line chart
st.line_chart(sample_data)

# Optionally show raw values
if st.checkbox("Show raw data"):
    st.write(sample_data)