import torch
from torch import nn


class TestNet(nn.Module):
    def __init__(self, L1, L2, n_components, n_hidden):
        super().__init__()

        self.d1 = n_components * 2 * L1
        self.d2 = n_components * 2 * L2

        self.lin1 = nn.Linear(self.d1, n_hidden + 1)
        self.lin2 = nn.Linear(n_hidden + self.d2, 3)

    def forward(self, o: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # o: <B, NS, D*2*L>
        # d: <B, NS, D*2*L>
        assert len(o.shape) == 3
        assert len(d.shape) == 3
        assert o.size(2) == self.d1
        assert d.size(2) == self.d2

        z1 = self.lin1(o)

        x = torch.cat((d, z1[..., 1:]), dim=2)

        sigma = z1[..., 0]  # <B, NS>
        rgb = self.lin2(x)  # <B, NS, 3>

        sigma = nn.functional.relu(sigma)
        rgb = nn.functional.sigmoid(rgb)

        return rgb, sigma



