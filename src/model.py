import torch
from torch import nn


class TestNet(nn.Module):
    def __init__(self, name: str, L1, L2, n_components, n_hidden):
        super().__init__()

        self.name = name

        self.d1 = n_components * 2 * L1 + n_components
        self.d2 = n_components * 2 * L2 + n_components

        self.lin1 = nn.Linear(self.d1, n_hidden + 1)
        self.lin2 = nn.Linear(n_hidden + self.d2, 3)

    def forward(
        self, o: torch.Tensor, d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class MediumNet(nn.Module):
    def __init__(self, name: str, L1, L2, n_components, n_hidden):
        super().__init__()

        self.name = name

        self.d1 = n_components * 2 * L1 + n_components
        self.d2 = n_components * 2 * L2 + n_components
        self.l5 = n_hidden + self.d2

        self.lin1 = nn.Linear(self.d1, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_hidden)
        self.lin4 = nn.Linear(n_hidden, n_hidden + 1)
        self.lin5 = nn.Linear(n_hidden + self.d2, n_hidden // 2)
        self.lin6 = nn.Linear(n_hidden // 2, 3)

    def forward(
        self, o: torch.Tensor, d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # o: <B, NS, D*2*L>
        # d: <B, NS, D*2*L>
        assert len(o.shape) == 3
        assert len(d.shape) == 3
        assert o.size(2) == self.d1
        assert d.size(2) == self.d2

        x = self.lin1(o)
        x = nn.functional.relu(x)
        x = self.lin2(x)
        x = nn.functional.relu(x)
        x = self.lin3(x)
        x = nn.functional.relu(x)
        x = self.lin4(x)

        # no activation func according to paper
        sigma = x[..., 0]  # <B, NS>
        x = torch.cat((d, x[..., 1:]), dim=2)
        x = self.lin5(x)
        x = nn.functional.relu(x)
        rgb = self.lin6(x)  # <B, NS, 3>

        sigma = nn.functional.relu(sigma)
        rgb = nn.functional.sigmoid(rgb)

        return rgb, sigma


class LargeNetNoView(nn.Module):
    def __init__(self, name: str, L1, L2, n_components, n_hidden, act_func=nn.ReLU):
        super().__init__()

        self.name = name

        self.d1 = n_components * 2 * L1 + n_components
        self.d2 = n_components * 2 * L2 + n_components
        self.l5 = n_hidden + self.d2

        self.lin1 = nn.Linear(self.d1, n_hidden)
        self.pre_stack = nn.Sequential(
            nn.Linear(self.d1, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden),
            act_func(),
        )

        self.post_stack = nn.Sequential(
            nn.Linear(n_hidden + self.d1, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden),
            act_func(),
            nn.Linear(n_hidden, n_hidden + 1),
        )

        self.final_stack = nn.Sequential(
            nn.Linear(n_hidden + self.d2, n_hidden // 2),
            act_func(),
            nn.Linear(n_hidden // 2, 3),
        )

    def forward(
        self, o: torch.Tensor, d: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # o: <B, NS, D*2*L>
        # d: <B, NS, D*2*L>
        assert len(o.shape) == 3
        assert len(d.shape) == 3
        assert o.size(2) == self.d1
        assert d.size(2) == self.d2

        x = self.pre_stack(o)
        x = torch.cat((o, x), dim=-1)
        x = self.post_stack(x)

        sigma = x[..., 0]
        x = torch.cat((d, x[..., 1:]), dim=-1)
        rgb = self.final_stack(x)

        sigma = nn.functional.relu(sigma)  # <B, NS>
        rgb = nn.functional.sigmoid(rgb)  # <B, NS, 3>

        return rgb, sigma
