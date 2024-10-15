import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from Model import UVModel, training
from Dataset import EUV_DATA
from datetime import datetime, timedelta
from typing import Tuple
from Functions import batches_to_sets
import torch


dates: Tuple[datetime, datetime] = [datetime(2021, 1, 1), datetime(2024, 10, 17)]
res = timedelta(hours= 1)
criterion = Adam
optimizer = torch.nn.functional.mse_loss
lr = 0.002
epochs = 100


model = UVModel.to('cuda')
EUV_Tensor = EUV_DATA(dates, res)
UVIndex = pd.read_csv("UV index - 2021-2024.csv")
UVIndex = UVIndex.drop("DATE", axis= "columns")
UVIndex = UVIndex.drop(index= 0)
UVIND_Tensor = Tensor(UVIndex.values)
train_set, val_set = batches_to_sets(EUV_Tensor, 24)
train_dl = DataLoader(train_set, 8, shuffle = False, pin_memory = True, num_workers = 8)
val_dl = DataLoader(val_set, 8, shuffle = False, pin_memory = True, num_workers = 8)
training(model, criterion, optimizer, epochs, train_dl, val_dl)

