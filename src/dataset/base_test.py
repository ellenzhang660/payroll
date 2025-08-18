from src.dataset.datasets.payroll_dataset import PayrollDataset
from src.dataset.datasets.generic_dataset import GenericDataset

def test_payroll():
    dataset = PayrollDataset()
    print(dataset)
    for i in range(dataset.how_many_unique_samples()):
        dataset[i]
    # assert str(dataset) == "hi"
    # assert repr(dataset) == "MyString('hi')"

def test_generic():
    dataset = GenericDataset()
    print(dataset)
    for i in range(dataset.how_many_unique_samples()):
        dataset[i]