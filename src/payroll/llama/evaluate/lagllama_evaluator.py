import os
import sys
from types import ModuleType
from typing import Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.model.forecast import Forecast
from lag_llama.gluon.estimator import LagLlamaEstimator
from pandas import Series
from torch.utils.data import Dataset
from tqdm import tqdm


class LagLlamaEvaluator:
    """
    Class for loading in a lagllama checkpoint and using it to evaluate a test set

    Args:
        checkpoint_path: path of the checkpoint
        prediction_length: prediciton length for the dataset, e.g. 6
        context_length: conext length for the dataset
        device: torch.device
        use_rope_scaling: should be True
        num_samples: number of samples to probabilistcally forecast
        save_dir: directory to save evaluation results

    Method
        _forecast(dataset):
            dataset is PandasDataset
            returns the forecast and target time series

        _visualize_forecast(forecast, ts):
            visualizes the forecast and target time series and saves it

         _calculate_metrics(self, forecast, ts):
            Returns
            -------

            agg_metrics: aggregated forecast metrics, we'll use the MAPE metric to evaluate
            ts_metrics: information about each time series, we'll use this to graph

        evaluate_test_dataset(dataset: Dataset):
            given a datase that returns a PandasDataset for a unique person/id,
                forecasts the time series
                visualizes the forecasts
                evaluates the forecast for uncertainty and quality on various forecasting metrics

    TODO:
        expand to evaluating for anamoly detection/other downstream tasks
    """

    def __init__(
        self,
        checkpoint_path: str,
        prediction_length: int,
        context_length: int,
        device: torch.device,
        use_rope_scaling: bool,
        num_samples: int,
        save_dir: Optional[str],
    ):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        }
        estimator = LagLlamaEstimator(
            ckpt_path=checkpoint_path,
            prediction_length=prediction_length,
            context_length=context_length,  # Lag-Llama was trained with a context length of 32, but can work with any context length
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
            batch_size=1,
            num_parallel_samples=100,
            device=device,
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        self.predictor = estimator.create_predictor(transformation, lightning_module)
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.evaluator = Evaluator()

    def _forecast(self, dataset: PandasDataset) -> Tuple[list[Forecast], list[Series]]:
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset, predictor=self.predictor, num_samples=self.num_samples
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        return forecasts, tss

    def _visualize_forecast(
        self, forecast: Forecast, ts: Series, metrics: Series, prediction_interval: Tuple[int, int] = (10, 90)
    ):
        """
        Visualizes a single forecast and its corresponding time series.

        Parameters:
        - forecast: GluonTS SampleForecast object
        - ts: pandas Series with a PeriodIndex or DateTimeIndex
        - prediction_interval: tuple of percentiles to show uncertainty (e.g., (10, 90))
        """

        def format_metric(key: str) -> str:
            """
            Returns metrics as a percentage
            """
            metric = float(f"{metrics[key]:.4f}") * 100.0
            metric = float(f"{metric:.2f}")
            return str(metric)

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

        plt.title(
            f"Start : {str(metrics["forecast_start"])} for item: {forecast.item_id} with MAPE: {format_metric("MAPE")}%"
        )
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()

        # Format x-axis
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

        plt.tight_layout()

        if self.save_dir:
            plt.savefig(f"{self.save_dir}/{forecast.item_id}")
            # logger.info(f"Saved forecast plot to: {save_path}")
        else:
            plt.show()
        plt.close()

    def _calculate_metrics(self, forecast: list[Forecast], ts: list[Series]) -> Tuple[dict[str, float], pd.DataFrame]:
        """
        Returns
        -------

        agg_metrics: aggregated forecast metrics, we'll use the MAPE metric to evaluate
            dictionary mapping metrics to its value
        ts_metrics: information about each time series, we'll use this to graph
            converts to a dataframe
        """
        agg_metrics, ts_metrics = self.evaluator(ts_iterator=ts, fcst_iterator=forecast)

        return agg_metrics, ts_metrics

    def evaluate_test_dataset(self, dataset: Dataset):
        if self.save_dir:
            os.makedirs(name=self.save_dir, exist_ok=True)
        for i in tqdm(range(len(dataset)), desc="Evaluating test set..."):
            dataframe = dataset[i]
            forecast, ts = self._forecast(dataset=dataframe)
            _, ts_metrics = self._calculate_metrics(forecast=forecast, ts=ts)
            self._visualize_forecast(forecast=forecast[0], ts=ts[0], metrics=ts_metrics.iloc[0])


"""
The below is a quick hack/fix for loading the checkpoint, as the NegativelogLikelihood is not working
Does not affect the evaluation of the model as we don't need a loss function anyways
"""


# Create dummy module hierarchy
def create_dummy_module(module_path):
    """
    Create a dummy module hierarchy for the given path.
    Returns the leaf module.
    """
    parts = module_path.split(".")
    current = ""
    parent = None

    for part in parts:
        current = current + "." + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = current

    return sys.modules[module_path]


# Create the dummy gluonts module hierarchy
gluonts_module = create_dummy_module("gluonts.torch.modules.loss")


# Create dummy classes for the specific loss functions
class DistributionLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class NegativeLogLikelihood:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


# Add the specific classes to the module
gluonts_module.DistributionLoss = DistributionLoss
gluonts_module.NegativeLogLikelihood = NegativeLogLikelihood
sys.modules["__main__"].NegativeLogLikelihood = NegativeLogLikelihood
