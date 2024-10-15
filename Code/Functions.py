from Dataset import EUV_DATA
from torch.utils.data import random_split
from torch import Tensor

def batches_to_sets(dataset: EUV_DATA, y_values: Tensor, batch_size: int):
    batches_num = dataset / batch_size
    batches = [dataset[i * batch_size:(i + 1) * batch_size] for i in range(batches_num)]
    if len(y_values) < len(batches):
        raise ValueError("Value tensor must have at least as many elements as there are batches.")
    train = []
    val = []
    split_index = round(len(batches) * 0.8)
    for i, batch in enumerate(batches):
        value = y_values[i].item()  # Extract value from tensor
        if i < split_index:
            train.append((Tensor(batch), value))
        else:
            val.append((Tensor(batch), value))
    return train, val

