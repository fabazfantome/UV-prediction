from torch import nn, Tensor
from typing import List
import torch
from torchvision.models import resnet18

## Ejemplo de arquitectura de modelo

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn: nn.Module = resnet18()

        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.rnn: nn.Module = nn.LSTM(input_size = 128, hidden_dim = 100, batch_first = True)

    def forward(self, x: Tensor) -> Tensor:
        _, frames, _, _, _ = x.size()
        cnn_out: List[Tensor] = []
        for t in range(frames):
            x_t = x[:, t, :, :, :]
            cnn_out.append(self.cnn(x_t))

        out: Tensor = torch.stack(cnn_out, dim = 1)

        out, _ = self.rnn(out) # si para el rnn porque ese vamos a entrenar

        return out






