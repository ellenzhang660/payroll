import argparse
from pathlib import Path

import torch

from src.payroll.llama.evaluate import init_dataset
from src.payroll.llama.evaluate.lagllama_evaluator import LagLlamaEvaluator


current_dir = Path(__file__).parent
"""
Evaluator for test set, runs LagLlama forecasting from a checkpoint and visualizes/evaluate the metrics

Args:
----
dataset
    dataset to evaluate on. 
from_dir
    directory the model is stored at, e.g.
    ${workspaceFolder}/lightning_logs
checkpoint_path
    the rest of the model path,
    eg. version_2/checkpoints/epoch=24-step=1250.ckpt

To run, cd into root repo and run
export PYTHONPATH=$(pwd)
poetry run python src/payroll/llama/scripts/evaluate.py --dataset payroll
"""


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["payroll", "generic", "weather"])
    parser.add_argument(
        "--from_dir", type=str, default="/mnt/c/Users/elzhang/Documents/payroll/model_checkpoints/lag-llama"
    )
    return parser.parse_args()


def find_ckpt_files(root: str, target_column: str) -> list[Path]:
    ckpt_dir = Path(root) / target_column / "lightning_logs"
    print(f'found checkpoints for {target_column}: {list(ckpt_dir.rglob("*.ckpt"))}')
    return list(ckpt_dir.rglob("*.ckpt"))


if __name__ == "__main__":
    args = set_args()
    save_dir = f"{current_dir}/results/{args.dataset}"
    dataset, target_columns = init_dataset(dataset=args.dataset)
    device = torch.device("cpu")
    for target_column in target_columns:
        ckpt = find_ckpt_files(args.from_dir, target_column)
        if len(ckpt) > 0:
            llama_forecast = LagLlamaEvaluator(
                checkpoint_path=str(ckpt[0]),
                prediction_length=dataset[target_column].attributes.prediction_length,
                context_length=dataset[target_column].attributes.context_length,
                device=device,
                use_rope_scaling=True,
                num_samples=10,
                save_dir=f"{save_dir}/{target_column}",
                target_column=target_column,
            )
            llama_forecast.evaluate_test_dataset(dataset=dataset[target_column])  # type: ignore
