# finetuning script from https://www.ibm.com/think/tutorials/lag-llama

import sys
from types import ModuleType

import torch
from gluonts.dataset.pandas import PandasDataset
from lag_llama.gluon.estimator import LagLlamaEstimator


"""
The below is a quick hack/fix for loading the checkpoint, as the NegativelogLikelihood is not working
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


class FinetuneLagLlama:
    """
    Fineune wrapper for Lagllama

    """

    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        device: torch.device,
        num_samples: int,
        batch_size: int,
        save_dir: str,
    ):
        ckpt = torch.load("lag-llama/lag-llama.ckpt", map_location=device, weights_only=False)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        self.finetune_estimator = LagLlamaEstimator(
            ckpt_path="lag-llama/lag-llama.ckpt",
            prediction_length=prediction_length,
            context_length=context_length,
            aug_prob=0.5,
            lr=5e-4,
            device=device,
            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],
            # linear positional encoding scaling
            rope_scaling={
                "type": "linear",
                "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            },
            batch_size=batch_size,
            num_parallel_samples=num_samples,
            trainer_kwargs={
                "max_epochs": 50,
                "default_root_dir": f"{save_dir}",  # path to save checkpoints/logs
            },  # lightning trainer arguments
        )
        self.num_samples = num_samples
        self.save_dir = save_dir

    def finetune(self, train: PandasDataset, valid: PandasDataset):
        self.finetuned_predictor = self.finetune_estimator.train(
            train, valid, cache_data=True, shuffle_buffer_length=1000
        )
        return self.finetuned_predictor
