from typing import Tuple, List
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import pandas as pd
from torch import Tensor
import torch
import numpy as np

def funcion_date(scrap_date: Tuple[datetime, datetime]) -> List[str]:
    current_date = scrap_date[0]
    last_date = scrap_date[-1]
    output = []
    while current_date <= last_date:
        output.append(current_date.strftime("%Y%m%d"))
        current_date+=timedelta(days = 1)
    return output

class EUV_DATA(Dataset):
    def __init__(self, scrap_date: Tuple[datetime, datetime]) -> None:
        self.dates = funcion_date(scrap_date)
        self.input_path = lambda date: f"./data/SDO/EVE/{date}.csv"

        self.inputs: List[Tensor] = []
        self.outputs: List[Tensor] = []

        self.output_csv = pd.read_csv("target.csv", parse_dates = True, index_col = 'DATE')

        for idx, prior in enumerate(self.dates[:-1]):
            try:
                self.inputs.append(torch.from_numpy(
                    pd.read_csv(self.input_path(prior))\
                    .replace(-1, np.nan)\
                    .dropna()\
                    .values
                ))
                self.outputs.append(torch.from_numpy(self.output_csv.iloc[idx + 1].values))
            except FileNotFoundError:
                continue

    def __len__(self) -> int:
        return len(self.inputs) - 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return torch.flatten(self.inputs[idx]), self.outputs[idx]

