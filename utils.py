import torch
import torch.nn as nn


def init_xavier_uniform(model: nn.Module):
    """
    Reference: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain("relu")
            )
            torch.nn.init.zeros_(m.bias)
