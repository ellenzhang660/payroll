"""
Zero shot forecasting for llama
not sure how this will work
TODO: synthetic data generation
"""

# from itertools import islice

# from matplotlib import pyplot as plt
# import matplotlib.dates as mdates

import sys
from types import ModuleType

import torch
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator


# from gluonts.dataset.repository.datasets import get_dataset

# from gluonts.dataset.pandas import PandasDataset
# import pandas as pd
# import numpy as np


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


class LlamaPredictions:
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        device: torch.device,
        use_rope_scaling: bool,
        num_samples: int,
    ):
        ckpt = torch.load(
            "lag-llama/lag-llama.ckpt", map_location=device, weights_only=False
        )  # Uses GPU since in this Colab we use a GPU.
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        print(estimator_args)

        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        }
        estimator = LagLlamaEstimator(
            ckpt_path="lag-llama/lag-llama.ckpt",
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

    def zero_shot(self, dataset):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset, predictor=self.predictor, num_samples=self.num_samples
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        return forecasts, tss
