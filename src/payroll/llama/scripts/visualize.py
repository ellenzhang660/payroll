# client selects specific person

# can visualize each of the forecasted plots for all available columns

from typing import Tuple

import streamlit as st
import torch
from matplotlib.figure import Figure

from src.payroll.llama.evaluate import init_dataset
from src.payroll.llama.evaluate.base_dataset import BaseTestDataset
from src.payroll.llama.evaluate.lagllama_evaluator import LagLlamaEvaluator
from src.payroll.llama.scripts.evaluate import find_ckpt_files


"""
PYTHONPATH=$(pwd) poetry run streamlit run src/payroll/llama/scripts/visualize.py
"""
device = torch.device("cpu")


# Cache dataset initialization
@st.cache_data
def load_dataset():
    dataset, target_columns = init_dataset(dataset="payroll")
    return dataset, target_columns


dataset, target_columns = load_dataset()
names = [f"Person_{i}" for i in range(len(dataset[target_columns[0]]))]


# Cache model loading
@st.cache_resource
def load_models() -> dict[str, Tuple[LagLlamaEvaluator, BaseTestDataset]]:
    root = "/mnt/c/Users/elzhang/Documents/payroll/model_checkpoints/lag-llama"
    llama_models: dict[str, Tuple[LagLlamaEvaluator, BaseTestDataset]] = {}
    for target_column in target_columns:
        if ckpt_files := find_ckpt_files(root, target_column):
            llama_models[target_column] = (
                LagLlamaEvaluator(
                    checkpoint_path=str(ckpt_files[0]),
                    prediction_length=dataset[target_column].attributes.prediction_length,
                    context_length=dataset[target_column].attributes.context_length,
                    device=device,
                    use_rope_scaling=True,
                    num_samples=10,
                    save_dir=None,
                    target_column=target_column,
                ),
                dataset[target_column],
            )
    return llama_models


llama_models = load_models()


@st.cache_data
# Function to generate a plot based on the name
def get_plots_for_person(name: str) -> dict[str, Figure]:
    idx = int(name.split("_")[-1])
    print(f"########### {idx} ###############")
    plots: dict[str, Figure] = {}
    for target_column in llama_models:
        try:
            evaluator, data = llama_models[target_column]
            dataframe = data.__getitem__(idx)
            plots[target_column] = evaluator.stream_evaluation(dataframe)
        except Exception as e:
            print(f"failed for {target_column} person {idx}")
            print(f"error {e}")
    return plots


# Streamlit UI
st.title("Time Series Forecasting")

selected_name = st.selectbox("Choose a person:", names)

if selected_name:
    st.write(f"You selected: {selected_name}")
    plots = get_plots_for_person(selected_name)
    st.write("Plotting...")
    for target_column, fig in plots.items():
        st.subheader(f"Plot for {target_column}")
        st.pyplot(fig)
