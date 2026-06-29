from __future__ import annotations

from dataclasses import dataclass


def _norm2d(torch, channels: int, normalization: str):
    nn = torch.nn
    if normalization == "layer_norm":
        return nn.GroupNorm(1, channels)
    return nn.Identity()


def build_fast_anogan_modules(torch, method_graph: dict, method_config: dict):
    """Build paper-near fastAnoGAN Generator, Critic, and Encoder modules.

    The critic uses per-sample GroupNorm(1, C) as a practical LayerNorm variant
    for 2D feature maps, avoiding BatchNorm's cross-sample coupling under WGAN-GP.
    """

    nn = torch.nn
    functional = torch.nn.functional

    class ResUpBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
            self.act = nn.LeakyReLU(0.2, inplace=False)

        def forward(self, x):
            up = functional.interpolate(x, scale_factor=2, mode="nearest")
            residual = self.skip(up)
            out = self.act(self.conv1(up))
            out = self.conv2(out)
            return self.act(out + residual)

    class ResDownBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, normalization: str = "none") -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.norm1 = _norm2d(torch, out_channels, normalization)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.norm2 = _norm2d(torch, out_channels, normalization)
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
            self.pool = nn.AvgPool2d(2)
            self.act = nn.LeakyReLU(0.2, inplace=False)

        def forward(self, x):
            residual = self.pool(self.skip(x))
            out = self.act(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            out = self.pool(out)
            return self.act(out + residual)

    class Generator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = method_graph["generator_blocks"]
            latent_dim = int(method_config["latent_dim"])
            seed_size = int(method_config.get("generator_seed_size", 4))
            seed_channels = int(blocks[0]["out_channels"])
            self.seed_size = seed_size
            self.seed_channels = seed_channels
            self.fc = nn.Linear(latent_dim, seed_channels * seed_size * seed_size)
            modules = []
            in_channels = seed_channels
            for block in blocks:
                out_channels = int(block["out_channels"])
                modules.append(ResUpBlock(in_channels, out_channels))
                in_channels = out_channels
            self.blocks = nn.Sequential(*modules)
            self.out = nn.Conv2d(in_channels, int(method_config["input_channels"]), 3, padding=1)
            self.activation = str(method_config.get("output_activation", "tanh"))

        def forward(self, z):
            x = self.fc(z).reshape(z.shape[0], self.seed_channels, self.seed_size, self.seed_size)
            x = self.blocks(x)
            x = self.out(x)
            if self.activation == "sigmoid":
                return torch.sigmoid(x)
            if self.activation == "tanh":
                return torch.tanh(x)
            return x

    class Critic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            modules = []
            in_channels = int(method_config["input_channels"])
            for block in method_graph["critic_blocks"]:
                normalization = str(block.get("normalization", "layer_norm"))
                out_channels = int(block["out_channels"])
                modules.append(ResDownBlock(in_channels, out_channels, normalization=normalization))
                in_channels = out_channels
            self.blocks = nn.Sequential(*modules)
            self.head = nn.LazyLinear(1)

        def features(self, x):
            return self.blocks(x)

        def forward(self, x):
            features = self.features(x)
            return self.head(features.flatten(1)).reshape(-1)

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            modules = []
            in_channels = int(method_config["input_channels"])
            for block in method_graph["encoder_blocks"]:
                out_channels = int(block["out_channels"])
                modules.append(ResDownBlock(in_channels, out_channels, normalization=str(block.get("normalization", "none"))))
                in_channels = out_channels
            self.blocks = nn.Sequential(*modules)
            self.head = nn.LazyLinear(int(method_config["latent_dim"]))
            self.activation = str(method_config.get("encoder_output_activation", "tanh"))

        def forward(self, x):
            z = self.head(self.blocks(x).flatten(1))
            if self.activation == "tanh":
                return torch.tanh(z)
            return z

    return Generator(), Critic(), Encoder()


@dataclass
class FastAnoganForward:
    reconstruction: object
    latent: object
    real_features: object
    reconstruction_features: object
    critic_real: object
    critic_reconstruction: object


def fast_anogan_forward(generator, critic, encoder, x) -> FastAnoganForward:
    z = encoder(x)
    reconstruction = generator(z)
    real_features = critic.features(x)
    reconstruction_features = critic.features(reconstruction)
    return FastAnoganForward(
        reconstruction=reconstruction,
        latent=z,
        real_features=real_features,
        reconstruction_features=reconstruction_features,
        critic_real=critic(x),
        critic_reconstruction=critic(reconstruction),
    )
