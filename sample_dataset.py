from torch.utils.data import Dataset
from torch import nn, Tensor
from numpy.typing import NDArray
from typing import List, Tuple
import pandas as pd
import os
import torch
from datetime import datetime, timedelta
from astropy.io import fits

def doy_to_ddyymm(year, doy):
    # Convert DOY to a date
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    # Format the date as DDMMYYYY
    formatted_date = date.strftime("%d%m%Y")
    return formatted_date

def read_img(path: str) -> Tuple[NDArray, datetime]:
    with fits.open(path) as hdul:
        img = hdul[0].data
        date = hdul[0].date
    return img, date

## NEW SAMPLE:
class SampleDataset(Dataset):
    def __init__(self, image_root_path: str, uv_index_py_path: str) -> None:
        self.image_path: List[str] = os.listdir(image_root_path)
        self.uv_index_py: pd.DataFrame = pd.read_csv(uv_index_py_path)

    def __len__(self) -> int:
        return len(self.image_path) - 24

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_paths: List[str] = self.image_path[idx : idx + 24]
        date_list: List[datetime] = []
        images: List[NDArray] = []

        for path in image_paths:
            img, date = read_img(path)
            images.append(img)
            date_list.append(date)

        init  = date_list[-1]
        end = init + timedelta(days = 4)

        uv = self.uv_index_py[(self.uv_index_py.index >= init) & (self.uv_index_py.index <= end)]
        uv = torch.from_numpy(uv.values)

        video: Tensor = torch.stack([torch.from_numpy(image) for image in images], dim = 1)

        return video, uv
