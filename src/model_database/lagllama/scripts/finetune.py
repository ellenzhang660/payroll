# finetuning script from https://www.ibm.com/think/tutorials/lag-llama

import argparse

import torch

from src.model_database.lagllama.finetune import init_dataset
from src.model_database.lagllama.finetune.lagllama_finetuner import FinetuneLagLlama


"""
Fineuner for lagllama on given dataset

Args
----
    dataset
        dataset to finetune lagllama on, of class BaseFinetuningDataset
    save_dir
        where to save the model checkpoints

Returns
-------
    Fineunes a lagllama model on all available target columns in the original dataset, saves each to a checkpoint 
    which can be loaded for evaluation

To run, cd into root repo and run
export PYTHONPATH=$(pwd)
poetry run python src/payroll/llama/finetune/finetune.py --dataset payroll
"""


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["payroll", "generic", "weather"])
    parser.add_argument("--save_dir", type=str, default="model_checkpoints/lag-llama")
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    datasets_dict = init_dataset(dataset=args.dataset)
    device = torch.device("cpu")
    num_samples = 20
    batch_size = 32
    for target_column, dataset in datasets_dict.items():
        try:
            finetune_llama = FinetuneLagLlama(
                prediction_length=dataset.prediction_length,
                context_length=dataset.context_length,
                device=device,
                num_samples=num_samples,
                target_column=target_column,
                batch_size=batch_size,
                save_dir=args.save_dir,
            )
            finetune_llama.finetune(train=dataset.train_dataset, valid=dataset.val_dataset)
        except Exception as e:
            print(f"error for {target_column}: {e}")
