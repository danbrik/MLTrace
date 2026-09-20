"""PyTorch implementations; imported lazily by workers and model checks."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class USAD(nn.Module):
    def __init__(self, length, channels, cfg):
        super().__init__()
        n = length * channels
        a, b = max(1, int(n * cfg["hidden_ratio_1"])), max(1, int(n * cfg["hidden_ratio_2"]))
        z = cfg["latent_dim"]
        self.encoder = nn.Sequential(nn.Linear(n, a), nn.ReLU(), nn.Linear(a, b), nn.ReLU(), nn.Linear(b, z), nn.ReLU())
        def decoder():
            return nn.Sequential(nn.Linear(z, b), nn.ReLU(), nn.Linear(b, a), nn.ReLU(), nn.Linear(a, n), nn.Sigmoid())
        self.decoder1, self.decoder2 = decoder(), decoder()

    def forward(self, x, sample=False):
        flat = x.flatten(1)
        z = self.encoder(flat)
        first = self.decoder1(z)
        second = self.decoder2(z)
        cascade = self.decoder2(self.encoder(first))
        return dict(reconstruction=first.reshape_as(x), second=second.reshape_as(x), cascade=cascade.reshape_as(x), latent=z)


class SameConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad // 2, self.pad - self.pad // 2)))


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel, dilation, dropout):
        super().__init__()
        self.main = nn.Sequential(SameConv(inputs, outputs, kernel, dilation), nn.ReLU(), nn.Dropout1d(dropout),
                                  SameConv(outputs, outputs, kernel, dilation), nn.ReLU(), nn.Dropout1d(dropout))
        self.shortcut = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x):
        skip = self.main(x)
        return F.relu(skip + self.shortcut(x)), skip


class TCN(nn.Module):
    def __init__(self, inputs, cfg):
        super().__init__()
        blocks = []
        for dilation in cfg["dilations"] * cfg["stacks"]:
            blocks.append(ResidualBlock(inputs, cfg["filters"], cfg["kernel_size"], dilation, cfg["dropout"]))
            inputs = cfg["filters"]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        skips = []
        for block in self.blocks:
            x, skip = block(x)
            skips.append(skip)
        return torch.stack(skips).sum(0)


class TCNAE(nn.Module):
    def __init__(self, length, channels, cfg):
        super().__init__()
        self.pool = cfg["pooling_factor"]
        if length < self.pool:
            raise ValueError("TCN-Poolingfaktor darf L nicht überschreiten. L wird nicht automatisch geändert.")
        self.encoder = TCN(channels, cfg)
        self.project = nn.Conv1d(cfg["filters"], cfg["latent_channels"], 1)
        self.decoder = TCN(cfg["latent_channels"], cfg)
        self.output = nn.Conv1d(cfg["filters"], channels, 1)

    def forward(self, x, sample=False):
        encoded = self.project(self.encoder(x.transpose(1, 2)))
        # ceil_mode includes a partial final bin and excludes nonexistent values.
        z = F.avg_pool1d(encoded, self.pool, self.pool, ceil_mode=True, count_include_pad=False)
        restored = z.repeat_interleave(self.pool, dim=2)[..., :x.shape[1]]
        reconstruction = self.output(self.decoder(restored)).transpose(1, 2)
        return dict(reconstruction=reconstruction, latent=z.transpose(1, 2).contiguous().flatten(1))


class LSTMVAE(nn.Module):
    def __init__(self, length, channels, cfg):
        super().__init__()
        h, z, layers = cfg["hidden_size"], cfg["latent_dim"], cfg["layers"]
        self.encoder = nn.LSTM(channels, h, layers, batch_first=True)
        self.z_mean, self.z_logvar = nn.Linear(h, z), nn.Linear(h, z)
        self.decoder = nn.LSTM(z, h, layers, batch_first=True)
        self.x_mean, self.x_var = nn.Linear(h, channels), nn.Linear(h, channels)

    def forward(self, x, sample=False):
        # No hidden state is passed or retained: each window is independent.
        encoded, _ = self.encoder(x)
        mu = self.z_mean(encoded)
        logvar = self.z_logvar(encoded).clamp(-20, 15)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        decoded, _ = self.decoder(z)
        variance = F.softplus(self.x_var(decoded)) + 1e-6
        return dict(reconstruction=self.x_mean(decoded), variance=variance, latent=mu[:, -1],
                    z_mean=mu, z_logvar=logvar)


def build_model(kind, length, channels, config):
    return {"usad": USAD, "tcn_ae": TCNAE, "lstm_vae": LSTMVAE}[kind](length, channels, config)


def errors(kind, x, output, training):
    if kind == "usad":
        alpha = training["alpha"]
        return alpha * (x - output["reconstruction"]).square() + (1 - alpha) * (x - output["cascade"]).square()
    if kind == "lstm_vae":
        variance = output["variance"]
        return 0.5 * (math.log(2 * math.pi) + variance.log() + (x - output["reconstruction"]).square() / variance)
    return (x - output["reconstruction"]).square()


def reconstruction_loss(kind, x, output, training):
    if kind == "tcn_ae":
        difference = output["reconstruction"] - x
        return (difference + F.softplus(-2 * difference) - math.log(2)).mean()
    nll = errors(kind, x, output, training).mean()
    if kind == "lstm_vae":
        kl = -0.5 * (1 + output["z_logvar"] - output["z_mean"].square() - output["z_logvar"].exp()).mean()
        return nll + training["kl_weight"] * kl
    return nll


def usad_losses(x, output, epoch):
    reconstruction1 = (x - output["reconstruction"]).square().mean()
    reconstruction2 = (x - output["second"]).square().mean()
    adversarial = (x - output["cascade"]).square().mean()
    return reconstruction1 / epoch + (1 - 1 / epoch) * adversarial, reconstruction2 / epoch - (1 - 1 / epoch) * adversarial



from app.time_series.definitions import representation_metadata
