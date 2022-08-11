import os
import zipfile
import requests 

import numpy as np

import torch
import torch.nn as nn

from ldm.models.autoencoder import AutoencoderKL
from ldm.models.autoencoder import VQModelInterface
from torch_utils.misc import get_cache_dir

# KL-f4
DEFAULT_AE_CONFIG = {
    "ddconfig": {
        "double_z": True,
        "z_channels": 3,
        "resolution": 256,
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
#VQ-f4
# DEFAULT_AE_CONFIG = {
#   "ddconfig": {
#     "double_z": False,
#     "z_channels": 3,
#     "resolution": 256,
#     "in_channels": 3,
#     "out_ch": 3,
#     "ch": 128,
#     "ch_mult": [1,2,4],
#     "num_res_blocks": 2,
#     "attn_resolutions": [],
#     "dropout": 0.0
#   },
#   "lossconfig": {
#         "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
#         "params": {
#           "disc_conditional": False,
#           "disc_in_channels": 3,
#           "disc_start": 0,
#           "disc_weight": 0.75,
#           "codebook_weight": 1.0
#         }
#   },
#   "n_embed": 8192,
#   "embed_dim": 3
# }
CACHE_MODEL_DIR = 'pretrained_models'
PREFIX = 'kl'
# PREFIX = 'vq'


#----------------------------------------------------------------------------
# GAN Normalising contanstants

STD_NORM = 70

#----------------------------------------------------------------------------
# Converts images to a latent space. Then converts the latent space to a suitable range for GANs (-1 to 1)

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_pre_trained_ae(url, prefix, folder):
    cache_dir = get_cache_dir()
    output_dir = f"{cache_dir}/{folder}"
    path = f"{output_dir}/{prefix}-model.ckpt"
    tmp_path = './tmp'
    if not os.path.exists(path):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        download_url(url, tmp_path)

        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        out_file = f'{output_dir}/model.ckpt'
        if os.path.exists(out_file):
            os.rename(out_file, path)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def setup():
    download_pre_trained_ae("https://ommer-lab.com/files/latent-diffusion/kl-f4.zip", PREFIX, CACHE_MODEL_DIR)
    # download_pre_trained_ae("https://ommer-lab.com/files/latent-diffusion/vq-f4.zip", PREFIX, CACHE_MODEL_DIR)

class Autoencoder:
    def __init__(self, device, type = 'kl'):
        self.device = device
        self.norm = STD_NORM

        print(f'Creating Autoencoder on device: {device}')
        model = AutoencoderKL(DEFAULT_AE_CONFIG["ddconfig"], DEFAULT_AE_CONFIG["lossconfig"], DEFAULT_AE_CONFIG["embed_dim"], ckpt_path=f"{get_cache_dir()}/{CACHE_MODEL_DIR}/{PREFIX}-model.ckpt")
        # model = VQModelInterface(embed_dim = DEFAULT_AE_CONFIG["embed_dim"], ddconfig = DEFAULT_AE_CONFIG["ddconfig"], lossconfig = DEFAULT_AE_CONFIG["lossconfig"], n_embed = DEFAULT_AE_CONFIG["n_embed"], ckpt_path=f"{get_cache_dir()}/{CACHE_MODEL_DIR}/{PREFIX}-model.ckpt")
        model = model.requires_grad_(False).to(device)

        # model.requires_grad_(True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)
        # model.requires_grad_(False)

        self._model = model

    # batch, channel, width, height
    def encode(self, images):
        with torch.no_grad():
            assert(len(images.shape) == 4)
            return self._model.module.encode(images).sample()
            # return self._model.encode(images)

    # batch, channel, width, height
    def decode(self, latent):
        with torch.no_grad():
            assert(len(latent.shape) == 4)
            latent = latent * self.norm
            return self._model.module.decode(latent)

#----------------------------------------------------------------------------
