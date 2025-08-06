import argparse
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.payroll.llama.dataset import init_dataset
from src.payroll.llama.llama import LlamaPredictions
from src.utils import logger


current_dir = Path(__file__).parent


def visualize_forecast(forecast, ts, save_path: str | None = None, prediction_interval=(10, 90)):
    """
    Visualizes a single forecast and its corresponding time series.

    Parameters:
    - forecast: GluonTS SampleForecast object
    - ts: pandas Series with a PeriodIndex or DateTimeIndex
    - save_path: optional path to save the figure instead of showing it
    - prediction_interval: tuple of percentiles to show uncertainty (e.g., (10, 90))
    """

    # Convert ts to timestamp if it's a PeriodIndex
    if isinstance(ts.index, pd.PeriodIndex):
        ts = ts.copy()
        ts.index = ts.index.to_timestamp()

    # Prepare forecast values
    forecast_index = pd.date_range(
        start=forecast.start_date.to_timestamp(), periods=forecast.samples.shape[1], freq=ts.index.freq or "M"
    )

    forecast_mean = forecast.mean
    lower, upper = np.percentile(forecast.samples, prediction_interval, axis=0)

    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(ts.index, ts.values, label="Target", linewidth=2)
    plt.plot(forecast_index, forecast_mean, color="green", label="Forecast (mean)")
    plt.fill_between(
        forecast_index,
        lower,
        upper,
        color="green",
        alpha=0.3,
        label=f"{prediction_interval[1] - prediction_interval[0]}% Prediction Interval",
    )

    plt.title(f"Forecast for item: {forecast.item_id}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/{forecast.item_id}")
        logger.info(f"Saved forecast plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["payroll", "generic"])
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    dataset = init_dataset(dataset=args.dataset)
    device = torch.device("cpu")
    llama_forecast = LlamaPredictions(
        prediction_length=dataset.prediction_length,
        context_length=dataset.context_length,
        device=device,
        use_rope_scaling=True,
        num_samples=10,
    )
    os.makedirs(name=f"{current_dir}/results/{args.dataset}", exist_ok=True)
    for i in tqdm(range(len(dataset)), desc=f"forecasting for {args.dataset}"):
        dataframe = dataset[i]
        forecast, ts = llama_forecast.zero_shot(dataset=dataframe)
        visualize_forecast(forecast=forecast[0], ts=ts[0], save_path=f"{current_dir}/results/{args.dataset}")
