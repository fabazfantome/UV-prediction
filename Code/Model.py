from torch import nn, Tensor
import torch

class UVModel(nn.Module):
    def __init__(self) -> None:
        super(UVModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(180, 360),
            nn.ReLU(),
            nn.Linear(360, 720),
            nn.ReLU(),
            nn.Linear(720, 720),
            nn.ReLU(),
            nn.Linear(720, 180),
            nn.ReLU(),
            nn.Linear(180, 12),
            nn.ReLU(),
            nn.Linear(12, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


def training(model, criterion, optimizer, lr, epochs, train_dl, val_dl):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optimizer(model.parameters(), lr)
    for epoch in range(epochs):
        train_history : list[Tensor] = []
        val_history : list[Tensor] = []
        for batch in train_dl:
            input, output = batch
            input = input.to(device)
            output = output.to(device)
            pred = model(input)
            loss = criterion(pred, output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(loss.detach())
        for batch in val_dl:
            input, output = batch
            input = input.to(device)
            output = output.to(device)
            pred = model(input)
            loss = criterion(pred, output)
            val_history.append(loss.detach())
        print(f'For epoch {epoch} training loss equals: {torch.stack(train_history).mean()}')
        print(f'For epoch {epoch} val loss equals: {torch.stack(val_history).mean()}')

