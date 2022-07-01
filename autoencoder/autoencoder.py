import numpy as np
import scipy.signal
import torch
from torch_utils import persistence
from torch_utils import misc
from torch_utils.ops import upfirdn2d
from torch_utils.ops import grid_sample_gradfix
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------
# GAN Normalising contanstants

norm = {
    "mean": 1,
    "std": 1,
}


#----------------------------------------------------------------------------
# Converts images to a latent space. Then converts the latent space to a suitable range for GANs (-1 to 1)

# @persistence.persistent_class
class Autoencoder(torch.nn.Module):
    def __init__(self,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
    ):
        super().__init__()

        # Pixel blitting.
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

    def forward(self, images, debug_percentile=None):
        assert isinstance(images, torch.Tensor) and images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        device = images.device
        if debug_percentile is not None:
            debug_percentile = torch.as_tensor(debug_percentile, dtype=torch.float32, device=device)
            

#----------------------------------------------------------------------------
