"""Reconstruction-only STAE for the 16 × 192 × 352 industrial-camera preset.

The graph is also used by Methods, training and inference. No prediction head,
dense projection or custom weight initialisation is applied.
"""


def reconstruction_graph():
    def layer(name, kind, **params):
        return {"id": name, "type": kind, "config": params}

    def block(prefix, kind, channels, **params):
        return [
            layer(prefix + "-conv", kind, out_channels=channels, kernel_size=3, bias=True, **params),
            layer(prefix + "-bn", "BatchNorm3d", num_features=channels, eps=1e-3, momentum=0.01),
            layer(prefix + "-act", "LeakyReLU", negative_slope=0.3, inplace=False),
        ]

    encoder, decoder = [], []
    for index, channels in enumerate([32, 48, 64, 64], 1):
        encoder.extend(block(f"enc-{index}", "Conv3d", channels, stride=1, padding=1))
        if index < 4:
            encoder.append(layer(f"enc-{index}-pool", "MaxPool3d", kernel_size=2, stride=2, padding=0))
    for index, channels in enumerate([48, 32, 32], 1):
        decoder.extend(block(f"dec-{index}", "ConvTranspose3d", channels, stride=2, padding=1, output_padding=1))
    decoder.append(layer("output", "Conv3d", out_channels=1, kernel_size=3, stride=1, padding=1, bias=True))
    return {"builder_kind": "spatiotemporal_autoencoder", "encoder": encoder,
            "latent": {"bottleneck_kind": "spatiotemporal", "shape": "64x2x24x44"},
            "decoder": decoder, "prediction_decoder": []}


def build_reconstruction_model(torch, graph=None, output_activation="sigmoid"):
    """Return a PyTorch module with tensor-valued forward() and encode().

    Torch remains optional for catalog/startup operations. Layers use PyTorch
    default initialisation; LazyConv3d infers the input channels on first use.
    """
    from app.modeling.forward import build_spatiotemporal_modules

    class STAE3DReconstruction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.decoder, _ = build_spatiotemporal_modules(torch, graph or reconstruction_graph())

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            y = self.decoder(z)
            if output_activation == "sigmoid":
                return torch.sigmoid(y)
            if output_activation == "tanh":
                return torch.tanh(y)
            return y

        def forward(self, x):
            return self.decode(self.encode(x))

    return STAE3DReconstruction()
