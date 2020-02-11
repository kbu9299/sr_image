import torch
import torch.nn as nn
import numpy as np


def lanczos_kernel(dx, a=3, N=None, dtype=None, device=None):
    """
    Generate Lanczos kernels for translation and interpolation
    """
    if not torch.is_tensor(dx):
        dx = torch.tensor(dx, dtype=dtype, device=device)

    if device is None:
        device = dx.device

    if dtype is None:
        dtype = dx.dtype

    D = dx.abs().ceil().int()
    S = 2 * (a + D) + 1  # width of kernel support

    S_max = S.max() if hasattr(S, 'shape') else S

    if (N is None) or (N < S_max):
        N = S

    Z = (N - S) // 2  # width of zeros beyond kernel support

    start = (-(a + D + Z)).min()
    end = (a + D + Z + 1).max()
    x = torch.arange(start, end, dtype=dtype, device=device).view(1, -1) - dx
    px = (np.pi * x) + 1e-3

    sin_px = torch.sin(px)
    sin_pxa = torch.sin(px / a)

    k = a * sin_px * sin_pxa / px ** 2  # sin(x) masked by sin(x/a)

    return k


def lanczos_shift(img, shift, p=3, a=3):
    """
    Shifts an image by convolving it with a Lanczos kernel
    """
    dtype = img.dtype

    if len(img.shape) == 2:
        img = img[None, None].repeat(1, shift.shape[0], 1, 1)  # batch of one image
    elif len(img.shape) == 3:  # one image per shift
        assert img.shape[0] == shift.shape[0]
        img = img[None,]

    # Apply padding
    padder = torch.nn.ReflectionPad2d(p)  # reflect pre-padding
    padded_img = padder(img)

    # Create 1D shifting kernels
    y_shift = shift[:, [0]]
    x_shift = shift[:, [1]]

    k_y = (lanczos_kernel(y_shift, a=a, N=None, dtype=dtype)
           .flip(1)  # flip axis of convolution
           )[:, None, :, None]  # expand dims to get shape (batch, channels, y_kernel, 1)
    k_x = (lanczos_kernel(x_shift, a=a, N=None, dtype=dtype)
           .flip(1)
           )[:, None, None, :]  # shape (batch, channels, 1, x_kernel)

    # Apply kernels
    shifted_img = torch.conv1d(padded_img,
                               groups=k_y.shape[0],
                               weight=k_y,
                               padding=[k_y.shape[2] // 2, 0])  # same padding
    shifted_img = torch.conv1d(shifted_img,
                               groups=k_x.shape[0],
                               weight=k_x,
                               padding=[0, k_x.shape[3] // 2])

    shifted_img = shifted_img[..., p:-p, p:-p]  # remove padding

    return shifted_img.squeeze()  # , k.squeeze()


class RegistrationModel(nn.Module):
    """
    Subpixel registration and interpolation with Lanczos kernel
    """

    def __init__(self, in_channel=1):
        super(RegistrationModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2 * in_channel, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.activ1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 2, bias=False)
        self.fc2.weight.data.zero_()  # init the weights with the identity transformation

    def forward(self, x):
        """
        Registers paris of images with sub-pixel shifts
        """
        x[:, 0] = x[:, 0] - torch.mean(x[:, 0], dim=(1, 2)).view(-1, 1, 1)
        x[:, 1] = x[:, 1] - torch.mean(x[:, 1], dim=(1, 2)).view(-1, 1, 1)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        out = out.view(-1, 128 * 16 * 16)
        out = self.drop1(out)  # dropout on spatial tensor (C*W*H)

        out = self.fc1(out)
        out = self.activ1(out)
        out = self.fc2(out)
        return out

    @staticmethod
    def transform(theta, img):
        """
        Shifts images by theta with Lanczos interpolation
        """
        shifted_img = lanczos_shift(img=img.transpose(0, 1), shift=theta.flip(-1), a=3, p=5)[:, None]
        return shifted_img
