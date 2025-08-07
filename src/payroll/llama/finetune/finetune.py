# finetuning script from https://www.ibm.com/think/tutorials/lag-llama

import argparse

import torch

from src.payroll.llama.finetune import init_dataset
from src.payroll.llama.finetune.lagllama_finetuner import FinetuneLagLlama


"""
Fineuner for lagllama on given dataset

To run, cd into root repo and run
export PYTHONPATH=$(pwd)
poetry run python src/payroll/llama/finetune/finetune.py --dataset payroll
"""


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["payroll", "generic", "weather"])
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    datasets_dict = init_dataset(dataset=args.dataset)
    device = torch.device("cpu")
    num_samples = 20
    batch_size = 64
    for target_column, dataset in datasets_dict.items():
        finetune_llama = FinetuneLagLlama(
            prediction_length=dataset.prediction_length,
            context_length=dataset.context_length,
            device=device,
            num_samples=num_samples,
            target_column=target_column,
            batch_size=batch_size,
        )
        finetune_llama.finetune(train=dataset.train_dataset, valid=dataset.val_dataset)
