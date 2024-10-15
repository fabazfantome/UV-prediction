from torch.utils.data import Dataset
from torch import Tensor
from datetime import datetime, timedelta
import starstream as st
from typing import Tuple


class EUV_DATA(Dataset):
    def __init__(self, scrap_date: Tuple[datetime, datetime], resolution: timedelta) -> None:
        self.satellite = st.SDO.EVE()
        self.data = self.satellite.get_torch(scrap_date, resolution)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int | slice) -> Tuple[Tensor, Tensor]:
        idx_hours = idx * 24
        return self.data[idx_hours - self : idx_hours, :]

