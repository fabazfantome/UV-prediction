from torch import nn, Tensor
import torch


class UVModel(nn.Module):
    def __init__ (self, input: int, output: int) -> None:
        super.__init__(
            nn.Linear(24*6, 24*12),
            nn.ReLU(),
            nn.Linear(24*12, 24^2),
            nn.ReLU(),
            nn.Linear(24^2, 24*6),
            nn.ReLU(),
            nn.Linear(24*6, 36),
            nn.ReLU(),
            nn.Linear(36, 9),
            nn.ReLU(),
            nn.Linear(9, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid
        )



def training(model, criterion, optimizer, lr, epochs, train_dl, val_dl):
    optimizer = optimizer(model.parameters(), lr = lr)
    for epoch in range(epochs):
        train_history : list[Tensor] = []
        val_history : list[Tensor] = []
        for batch in train_dl:
            out = model(batch[0].to('cuda'))
            loss = criterion(out.squeeze(-1), batch[1].to(torch.float).to('cuda'))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(loss.detach())
        for batch in val_dl:
            out = model(batch[0].to('cuda'))
            loss = criterion(out.squeeze(-1), batch[1].to(torch.float).to('cuda'))
            val_history.append(loss.detach())
        print(f'For epoch {epoch} training loss equals: {torch.stack(train_history).mean()}')
        print(f'For epoch {epoch} val loss equals: {torch.stack(val_history).mean()}')

