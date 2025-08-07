import argparse
import os
from pathlib import Path

import torch

from src.payroll.llama.evaluate.lagllama_evaluator import LagLlamaEvaluator
from src.payroll.llama.evaluate.test_datasets import init_dataset


current_dir = Path(__file__).parent
"""
Evaluator for test set, runs LagLlama forecasting from a checkpoint and visualizes/evaluate the metrics

To run, cd into root repo and run
export PYTHONPATH=$(pwd)
poetry run python src/payroll/llama/evaluate/evaluate.py --dataset payroll
"""


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["payroll", "generic", "weather"])
    parser.add_argument("--from_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    save_dir = f"{current_dir}/results/{args.dataset}/{args.checkpoint_path}"
    checkpoint_path = os.path.join(args.from_dir, args.checkpoint_path)
    dataset = init_dataset(dataset=args.dataset)
    device = torch.device("cpu")
    llama_forecast = LagLlamaEvaluator(
        checkpoint_path=checkpoint_path,
        prediction_length=dataset.prediction_length,
        context_length=dataset.context_length,
        device=device,
        use_rope_scaling=True,
        num_samples=10,
        save_dir=save_dir,
    )
    llama_forecast.evaluate_test_dataset(dataset=dataset)
