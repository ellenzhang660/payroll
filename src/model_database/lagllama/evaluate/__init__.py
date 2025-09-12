from typing import Literal, Tuple

from src.model_database.lagllama.evaluate.base_dataset import BaseTestDataset
from src.model_database.lagllama.evaluate.test_datasets import GenericDataset, PayrollDataset


def init_dataset(dataset: Literal["payroll", "generic"]) -> Tuple[dict[str, BaseTestDataset], Tuple[str, ...]]:
    """
    Given daaset, returns a dictionary mapping
        key: target_column
        val: class FinetuneDataset(ClassAttributes):
            train_dataset: PandasDataset
            val_dataset: PandasDataset
    """
    if dataset == "payroll":
        base = PayrollDataset(target_column="Gross pay")
        mode = PayrollDataset
    elif dataset == "generic":
        base = GenericDataset(target_column="target")
        mode = GenericDataset
    else:
        raise ValueError

    datasets: dict[str, BaseTestDataset] = {}
    for target_column in base.available_target_columns:
        datasets[target_column] = mode(target_column=target_column)
    return datasets, base.available_target_columns
