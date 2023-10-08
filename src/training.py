import torch
from torch import nn


def expected_color(c, sigma, dt):
    # c: <B, N, 3>
    # sigma: <B, NS>
    # delta: <B, NR, NB>
    # B: batch size
    # N: number of samples
    # C: number of components

    assert len(sigma.shape) == 2
    assert len(dt.shape) == 3
    assert len(c.shape) == 3

    B, NR, NB = dt.shape
    NB = NB + 1
    C = c.size(-1)

    # Unpack from n_samples to n_rays x n_bins
    sigma = sigma.reshape(B, NR, NB)
    c = c.reshape(B, NR, NB, C)

    mul = dt * sigma[..., :-1]

    # Compute cumuluative probability,
    # Since equation (3) sums T_i from i=1 to i-1, we set the first value to (exp 0 = 1) and ignore the last value
    T = torch.exp(-torch.cumsum(mul, dim=-1))
    T = torch.cat((torch.ones(B, NR, 1), T), dim=-1)[..., :-1]

    # Since we do no have a delta for the last value,
    # we directly set the last value of w to T at i=N,
    # which is the dot product between sigma and delta
    T_N = torch.einsum("brn,brn->br", dt, sigma[..., :-1])[..., None]
    w = T * (1 - torch.exp(-mul))
    w = torch.cat((w, T_N), dim=-1)

    c_hat = torch.einsum("brn,brnc->brc", w, c)
    return c_hat


def positional_encoding(p: torch.Tensor, L: int) -> torch.Tensor:
    assert len(p.shape) == 4
    B, NR, NB, D = p.shape
    # p = p.reshape(B, NR*NB, D)

    # Z denotes transformed input p
    # Z_ij becomes 2^i * p_i * p_j for each i in 0..L-1 and each component j in 1..3
    # Thus dimension is <B, D, L>
    z = (2 ** torch.arange(L).repeat(D, 1)) * (torch.pi * p[..., None])

    # X denotes the encoded value for each transformed input
    x1 = torch.sin(z)
    x2 = torch.cos(z)

    # We want ordering sin(x) cos(x) sin(y) cos(y) sin(z) cos(z) repeated for each element in 1..L
    # First we stack encoding into a matrix, then we flatten the matrix to put each row side by side.
    x = torch.stack((x1, x2), dim=5)  # <B, NR, NB, D, L, 2>
    x = x.swapaxes(3, 4)  # <B, NR, NB, L, D, 2>
    x = x.reshape(B, NR * NB, 2 * D * L)  # Finally, flatten to shape <B, N, 2*D*L>

    return x


def strat_sampling(N: int, t_near: float, t_far: float) -> torch.Tensor:
    samples = (torch.arange(N) + torch.rand(N)) * (t_far - t_near) / N  # <N>
    return samples


def get_t(batch_size, n_rays, n_bins, t_near, t_far) -> torch.Tensor:
    t = strat_sampling(batch_size * n_rays * n_bins, t_near, t_far).reshape(
        batch_size, n_rays, n_bins
    )
    dt = torch.diff(t, dim=-1)
    return t, dt


import pytorch_lightning as ptl


class LitNerf(ptl.LightningModule):
    def __init__(
        self,
        scene_model: nn.Module,
        n_rays,
        n_bins,
        t_near,
        t_far,
        L1,
        L2,
        learning_rate: float = 3e-4,
    ):
        super().__init__()
        self.scene_model = scene_model
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.n_rays = n_rays
        self.n_bins = n_bins
        self.t_near = t_near
        self.t_far = t_far
        self.L1 = L1
        self.L2 = L2

    def training_step(self, batch, batch_idx):
        r_o, r_d, C_r = batch

        B = r_o.size(0)
        t, dt = get_t(B, self.n_rays, self.n_bins, self.t_near, self.t_far)

        r_d = nn.functional.normalize(r_d, dim=-1)

        r_o = r_o.reshape(B, -1, 1, 3)
        r_d = r_d.reshape(B, -1, 1, 3)
        r_d = r_d.repeat(1, 1, self.n_bins, 1)
        t = t[..., None]

        C_r = C_r[:, :3].reshape(B, 3, -1).swapaxes(1, 2)

        x = r_o + t * r_d
        ex = positional_encoding(x, self.L1)
        ed = positional_encoding(r_d, self.L2)

        c, sigma = self.scene_model(ex, ed)
        c_hat = expected_color(c, sigma, dt)

        loss = self.criterion(c_hat, C_r)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
