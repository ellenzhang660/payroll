from typing import Literal

from src.model_database.lagllama.finetune.base_dataset import FinetuneDataset
from src.model_database.lagllama.finetune.created_datasets import PayrollDataset, WeatherDataset


def init_dataset(dataset: Literal["payroll", "weather"]) -> dict[str, FinetuneDataset]:
    """
    Given daaset, returns a dictionary mapping
        key: target_column
        val: class FinetuneDataset(ClassAttributes):
            train_dataset: PandasDataset
            val_dataset: PandasDataset
    """
    if dataset == "payroll":
        base = PayrollDataset()
    elif dataset == "weather":
        base = WeatherDataset()

    datasets: dict[str, FinetuneDataset] = {}
    for target_column in base.available_target_columns:
        datasets[target_column] = base[target_column]
    return datasets
