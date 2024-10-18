import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from Model import UVModel, training
from Dataset import EUV_DATA
from datetime import datetime
from typing import Tuple
import torch


dates: Tuple[datetime, datetime] = [datetime(2021, 1, 1), datetime(2024, 10, 17)]
criterion = Adam
optimizer = torch.nn.functional.mse_loss
lr = 0.002
epochs = 100
path = "./data/SDO/EVE/"
dates = (datetime(2021, 1, 1), datetime(2024, 6, 16))
dataset = EUV_DATA(dates)


model = UVModel()
train_len = round(len(dataset)*0.8)
val_len = len(dataset) - train_len
train, val = random_split(dataset, [train_len, val_len])
train_dl = DataLoader(train, 8, shuffle = False, pin_memory = True, num_workers = 8)
val_dl = DataLoader(val, 8, shuffle = False, pin_memory = True, num_workers = 8)
training(model, criterion, optimizer, lr, epochs, train_dl, val_dl)

