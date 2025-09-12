import os
import pickle
import random
from typing import Literal

import torch
from lag_llama.gluon.estimator import LagLlamaEstimator
from torch.utils.data import Dataset

from src.model_database import model_database
from src.model_database.base import ModelChoice
from src.model_database.lagllama.evaluate.test_datasets import PayrollDataset
from src.model_database.lagllama.finetune.lagllama_finetuner import FinetuneLagLlama


current_working_dir = os.getcwd()
from datetime import datetime

import pandas as pd
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LagLlama(ModelChoice):
    """
    LagLlama wrapper
    """

    checkpoint_path = f"{current_working_dir}/lag-llama/lag-llama.ckpt"

    def __init__(self):
        super().__init__(model=model_database["lag-llama"])

        ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        self.estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    def zero_shot(self, data: Dataset, context_length: int, prediction_length: int):
        """
        Zero shot for LagLlama
        """
        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / self.estimator_args["context_length"]),
        }
        estimator = LagLlamaEstimator(
            ckpt_path=self.checkpoint_path,
            prediction_length=prediction_length,
            context_length=context_length,  # Lag-Llama was trained with a context length of 32, but can work with any context length
            # estimator args
            input_size=self.estimator_args["input_size"],
            n_layer=self.estimator_args["n_layer"],
            n_embd_per_head=self.estimator_args["n_embd_per_head"],
            n_head=self.estimator_args["n_head"],
            scaling=self.estimator_args["scaling"],
            time_feat=self.estimator_args["time_feat"],
            rope_scaling=rope_scaling_arguments,
            batch_size=1,
            num_parallel_samples=100,
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self.predictor = estimator.create_predictor(transformation, lightning_module)
        forecasts = self.zero_shot_forecast(dataset=data)
        print(forecasts)

    def zero_shot_forecast(self, dataset):
        all_rows = []
        for i in tqdm(range(len(dataset)), desc="Processing"):
            forecast_it = self.predictor.predict(dataset[i])
            forecast = list(forecast_it)
            predicted_forecast = forecast[0].samples[0]  # array of length N
            item_id = forecast[0].item_id
            start_date = forecast[0].start_date  # Period('2025-01', 'M')

            # Convert start_date to pandas Period if not already
            if not isinstance(start_date, pd.Period):
                start_date = pd.Period(start_date, freq="M")

            # Generate column names for each time step
            time_columns = [(start_date + j).strftime("%Y-%m") for j in range(len(predicted_forecast))]

            # Create row dictionary
            row = {"id": item_id}
            row.update({col: val for col, val in zip(time_columns, predicted_forecast)})

            all_rows.append(row)
        return all_rows

    def finetune(self, data, context_length: int, prediction_length: int):
        """Returns the number of samples of the time series data, assuming that each person + variable combination is the same"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        finetune = FinetuneLagLlama(
            prediction_length=prediction_length,
            context_length=context_length,
            device=device,
            num_samples=100,
            batch_size=32,
            save_dir=f"{current_working_dir}/model_checkpoints/lag-llama/{timestamp}",
        )
        finetune.finetune(train=data, valid=data)


class SynData(Dataset):
    def __init__(self, type: Literal["syn", "ori"]):
        super().__init__()
        root = f"{current_working_dir}/src/database/TimeGAN/payroll"
        data_path = os.path.join(root, "original_data.pkl")
        synthetic_path = os.path.join(root, "generated_data.pkl")

        # Load original data
        with open(data_path, "rb") as f:
            ori_data = pickle.load(f)
        self.ori_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in ori_data]

        # Load synthetic data
        with open(synthetic_path, "rb") as f:
            syn_data = pickle.load(f)
        self.syn_data = [arr.squeeze(-1) if arr.shape[-1] == 1 else arr for arr in syn_data]

        self.type = type

    def __len__(self):
        return len(self.ori_data) if self.type == "ori" else len(self.syn_data)

    def __getitem__(self, idx: int):
        start_year = random.randint(2020, 2024)
        start_month = random.randint(1, 12)
        start_period = pd.Period(f"{start_year}-{start_month:02d}", freq="M")
        if self.type == "ori":
            return self.ori_data[idx]
        elif self.type == "syn":
            return {"start": start_period, "target": self.syn_data[idx]}
        else:
            raise ValueError


if __name__ == "__main__":

    original = PayrollDataset(target_column="Gross pay")
    synthetic = SynData(type="syn")
    lagllama = LagLlama()
    # lagllama.zero_shot(original, context_length=24, prediction_length=12)
    lagllama.finetune(synthetic, context_length=24, prediction_length=12)
