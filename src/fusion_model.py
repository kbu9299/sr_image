import torch
import torch.nn as nn


class ResidualLayer(nn.Module):
    """
    Residual Layer
    """

    def __init__(self, channel_size, kernel_size):
        super(ResidualLayer, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual


class EncodeLayer(nn.Module):
    """
    Encode Layer
    """

    def __init__(self, in_channels=2, num_layers=2, kernel_size=3, channel_size=64):
        super(EncodeLayer, self).__init__()
        padding = kernel_size // 2
        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())
        res_layers = [ResidualLayer(channel_size, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class RecursiveLayer(nn.Module):
    """
    Recursive Layer
    """

    def __init__(self, in_channels=64, kernel_size=3):
        super(RecursiveLayer, self).__init__()
        self.alpha_residual = True

        self.fuse = nn.Sequential(
            ResidualLayer(2 * in_channels, kernel_size),
            nn.Conv2d(in_channels=2 * in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.PReLU())

    def forward(self, x):
        batch_size, seq_len, channels, width, height = x.shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        alphas = torch.ones([8], dtype=torch.float32, device=device)
        alphas = alphas.view(1, -1, 1, 1, 1)

        parity = seq_len % 2
        half_len = seq_len // 2

        while half_len > 0:
            alice = x[:, :half_len]  # first half hidden states (B, L/2, C, W, H)
            bob = x[:, half_len:seq_len - parity]  # second half hidden states (B, L/2, C, W, H)
            bob = torch.flip(bob, [1])

            alice_and_bob = torch.cat([alice, bob], 2)  # concat hidden states across channels (B, L/2, 2*C, W, H)
            alice_and_bob = alice_and_bob.view(-1, 2 * channels, width, height)
            x = self.fuse(alice_and_bob)
            x = x.view(batch_size, half_len, channels, width, height)  # new hidden states (B, L/2, C, W, H)

            alphas_alice = alphas[:, :half_len]
            alphas_bob = alphas[:, half_len:seq_len - parity]
            alphas_bob = torch.flip(alphas_bob, [1])
            x = alice + alphas_bob * x
            alphas = alphas_alice
            seq_len = half_len
            parity = seq_len % 2
            half_len = seq_len // 2

        return torch.mean(x, 1)


class DecodeLayer(nn.Module):
    """
    Decode Layer
    """

    def __init__(self):
        super(DecodeLayer, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=3),
            nn.PReLU())
        self.final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.deconv(x)
        x = self.final(x)
        return x


class FusionModel(nn.Module):
    """
    FusionModel: generate a Super-Resolved image from multiple low-resolution images
    """

    def __init__(self):
        super(FusionModel, self).__init__()
        self.encode = EncodeLayer()
        self.recursive = RecursiveLayer()
        self.decode = DecodeLayer()

    def forward(self, lrs):
        batch_size, seq_len, height, width = lrs.shape
        lrs = lrs.view(-1, seq_len, 1, height, width)
        refs, _ = torch.median(lrs[:, :seq_len], 1, keepdim=True)  # anchor shared multiple views
        refs = refs.repeat(1, seq_len, 1, 1, 1)
        stacked_input = torch.cat([lrs, refs], 2)  # tensor (B, L, 2*C_in, W, H)
        stacked_input = stacked_input.view(batch_size * seq_len, 2, width, height)
        layer1 = self.encode(stacked_input)  # encode input tensor
        layer1 = layer1.view(batch_size, seq_len, -1, width, height)  # tensor (B, L, C, W, H)
        recursive_layer = self.recursive(layer1)  # fuse hidden states (B, C, W, H)
        srs = self.decode(recursive_layer)  # decode final hidden state (B, C_out, 3*W, 3*H)
        return srs
