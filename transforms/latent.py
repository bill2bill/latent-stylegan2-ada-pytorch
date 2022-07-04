import os
import wget
import zipfile

import numpy as np

import torch
import torch.nn as nn

from ldm.models.autoencoder import AutoencoderKL

DEFAULT_AE_CONFIG = {
    "ddconfig": {
        "double_z": True,
        "z_channels": 3,
        "resolution": 64,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1,2,4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0
    },
    "lossconfig": {
        "target": "ldm.modules.losses.LPIPSWithDiscriminator",
        "params": {
            "disc_start": 50001,
            "kl_weight": 1.0e-06,
            "disc_weight": 0.5
        }
    },
    "embed_dim": 3
}
CACHE_MODEL_DIR = 'pretrained_models'

#----------------------------------------------------------------------------
# GAN Normalising contanstants

norm = {
    "mean": 0,
    "std": 80,
}

#----------------------------------------------------------------------------
# Converts images to a latent space. Then converts the latent space to a suitable range for GANs (-1 to 1)

def download_pre_trained_ae(url, output_dir):
    if os.path.exists(f"{CACHE_MODEL_DIR}/model.ckpt"):
        print("Used cache")
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = wget.download(url)

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        if os.path.exists(filename):
            os.remove(filename)

def setup():
    download_pre_trained_ae("https://ommer-lab.com/files/latent-diffusion/kl-f4.zip", CACHE_MODEL_DIR)


class Autoencoder():
    def __init__(self, device, num_gpus):
        self.device = device
        pl_sd = torch.load(f"{CACHE_MODEL_DIR}/model.ckpt")
        model = AutoencoderKL(DEFAULT_AE_CONFIG["ddconfig"], DEFAULT_AE_CONFIG["lossconfig"], DEFAULT_AE_CONFIG["embed_dim"])
        model.load_state_dict(pl_sd["state_dict"] ,strict=False)
        model.to(device)
        if (num_gpus > 1):
            model.encoder = nn.DataParallel(model.encoder, list(range(num_gpus)))
            model.decoder = nn.DataParallel(model.decoder, list(range(num_gpus)))
        self.model = model

    def encode(self, images):
        latent = self.model.encode(images.to(self.device)).sample()
        norm_latent = latent / norm['std']
        return torch.clamp(norm_latent, -1., 1.)

    def decode(self, norm_latent):
        latent = norm_latent.to(self.device) * norm['std']
        return self.model.decode(latent)

#----------------------------------------------------------------------------
