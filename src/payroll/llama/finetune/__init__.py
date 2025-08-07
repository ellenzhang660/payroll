from typing import Literal

from src.payroll.llama.finetune.base_dataset import FinetuneDataset
from src.payroll.llama.finetune.dataset import PayrollDataset, WeatherDataset


def init_dataset(dataset: Literal["payroll", "weather"], mode: Literal["train", "val", "test"]) -> FinetuneDataset:
    if dataset == "payroll":
        return PayrollDataset(target_column="Gross total")[mode]
    elif dataset == "weather":
        return WeatherDataset()[mode]
