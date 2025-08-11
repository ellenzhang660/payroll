from typing import Literal

from src.payroll.llama.evaluate.base_dataset import BaseTestDataset
from src.payroll.llama.evaluate.test_datasets import GenericDataset, PayrollDataset


def init_dataset(dataset: Literal["payroll", "generic"]) -> dict[str, BaseTestDataset]:
    """
    Given daaset, returns a dictionary mapping
        key: target_column
        val: class FinetuneDataset(ClassAttributes):
            train_dataset: PandasDataset
            val_dataset: PandasDataset
    """
    if dataset == "payroll":
        base = PayrollDataset(target_column="Gross pay")
    elif dataset == "generic":
        base = GenericDataset(target_column="target")
    else:
        raise ValueError

    datasets: dict[str, BaseTestDataset] = {}
    for target_column in base.available_target_columns:
        datasets[target_column] = base[target_column]
    return datasets
