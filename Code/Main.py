import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from Model import UVModel, training
from Dataset import EUV_DATA
from datetime import datetime, timedelta
from typing import Tuple
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dates: Tuple[datetime, datetime] = [datetime(2021, 1, 1), datetime(2024, 6, 16)]
criterion = torch.nn.functional.mse_loss
optimizer = Adam
lr = 0.001
epochs = 1000
path = "./data/SDO/EVE/"
dates = (datetime(2021, 1, 1), datetime(2024, 6, 16))
dataset = EUV_DATA(dates)


model = UVModel().to(torch.float32).to(device)
train_len = round(len(dataset)*0.8)
val_len = len(dataset) - train_len
train, val = random_split(dataset, [train_len, val_len])
train_dl = DataLoader(train, 16, shuffle = True)
val_dl = DataLoader(val, 16, shuffle = False)
training(model, criterion, optimizer, lr, epochs, train_dl, val_dl)

