from .datasets import Dataset


def load_dataset(*args, **kwargs):
    dataset = Dataset()
    dataset.load_dataset(*args, **kwargs)
    return dataset
