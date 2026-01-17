import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


class TinyNerfModel(nn.Module):
    def __init__(self, W=256, L_embed=6):
        super().__init__()

        self.L_embed = L_embed
        input_size = 3 + 2 * 3 * L_embed

        self.block1 = nn.Sequential(
            nn.Linear(input_size, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(W + input_size, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, W),
            nn.ReLU(),
            nn.Linear(W, 4),
        )

    def posenc(self, x: Float[Tensor, "B 3"]) -> Float[Tensor, "B 3"]:
        output = [x]
        for i in range(self.L_embed):
            output.append(torch.sin(2.0**i * x))
            output.append(torch.cos(2.0**i * x))

        return torch.cat(output, dim=-1)

    def forward(
        self, inputs: Float[Tensor, "B 3"]
    ) -> tuple[Float[Tensor, "B 3"], Float[Tensor, "B 1"]]:
        """
        inputs: (batch_dim, 3)
        returns: RGB -> (batch_dim, 3), Sigma -> (batch_dim, 1)
        """
        inputs = self.posenc(inputs)
        a = self.block1(inputs)
        b = self.block2(torch.cat([a, inputs], dim=-1))
        return F.sigmoid(b[..., :3]), F.relu(b[..., 3])
