import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNerfModel(nn.Module):
    def __init__(self, W=256):
        super().__init__()
        
        input_size = 3
        
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
    

    def forward(self, inputs):
        """
            inputs: (batch_dim, 3)
            returns: RGB -> (batch_dim, 3), Sigma -> (batch_dim, 1)
        """
        a = self.block1(inputs)
        b = self.block2(torch.cat([a, inputs], dim=-1))
        return  F.sigmoid(b[..., :3]), F.relu(b[...,3])