from typing import Tuple

import torch
from tqdm import tqdm

from src.payroll.llama.evaluate import init_dataset
from src.payroll.llama.evaluate.base_dataset import BaseTestDataset
from src.payroll.llama.evaluate.lagllama_evaluator import LagLlamaEvaluator
from src.payroll.llama.scripts.evaluate import find_ckpt_files
import pandas as pd

device = torch.device("cpu")

"""
PYTHONPATH=$(pwd) poetry run python src/model_database/llama/scripts/store.py
"""


# Cache dataset initialization
def load_dataset():
    dataset, target_columns = init_dataset(dataset="payroll")
    return dataset, target_columns


dataset, target_columns = load_dataset()
names = [f"Person_{i}" for i in range(len(dataset[target_columns[0]]))]


# Cache model loading
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
                    num_samples=1,
                    save_dir=None,
                    target_column=target_column,
                ),
                dataset[target_column],
            )
    return llama_models


llama_models = load_models()


all_rows = []

for target_column in llama_models:
    evaluator, dataset = llama_models[target_column]
    
    for i in tqdm(range(len(dataset)), desc=f"Processing {target_column}"):
        try:
            forecast = evaluator.get_forecast_for_unseen(dataset[i])
            predicted_forecast = forecast[0].samples[0]  # array of length N
            item_id = forecast[0].item_id
            start_date = forecast[0].start_date  # Period('2025-01', 'M')
            
            # Convert start_date to pandas Period if not already
            if not isinstance(start_date, pd.Period):
                start_date = pd.Period(start_date, freq='M')
            
            # Generate column names for each time step
            time_columns = [(start_date + j).strftime('%Y-%m') for j in range(len(predicted_forecast))]
            
            # Create row dictionary
            row = {'id': item_id, 'target_column': target_column}
            row.update({col: val for col, val in zip(time_columns, predicted_forecast)})
            
            all_rows.append(row)
        
        except Exception as e:
            print(f"Error {e} for {target_column}, index {i}")

# Convert to DataFrame
df = pd.DataFrame(all_rows)

# Save to CSV
df.to_csv('forecast_results.csv', index=False)
