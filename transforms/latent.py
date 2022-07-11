import os
import zipfile
import requests 

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



def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_pre_trained_ae(url, output_dir):
    path = f"{CACHE_MODEL_DIR}/model.ckpt"
    tmp_path = './tmp'
    if os.path.exists(path):
        print("Used cache")
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        download_url(url, tmp_path)

        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def setup():
    download_pre_trained_ae("https://ommer-lab.com/files/latent-diffusion/kl-f4.zip", CACHE_MODEL_DIR)

class Autoencoder:
    def __init__(self, rank = 0):
        device = torch.device('cuda', rank)
        self.device = device

        pl_sd = torch.load(f"{CACHE_MODEL_DIR}/model.ckpt")
        print(f'Creating Autoencoder on rank: {rank}')
        model = AutoencoderKL(DEFAULT_AE_CONFIG["ddconfig"], DEFAULT_AE_CONFIG["lossconfig"], DEFAULT_AE_CONFIG["embed_dim"])
        model.load_state_dict(pl_sd["state_dict"] ,strict=False)
        model.to(device)

        modules = [model.quant_conv, model.post_quant_conv, model.encoder, model.decoder]

        for module in modules:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        self._model = model

    # batch, channel, width, height
    def encode(self, images):
        with torch.no_grad():
            assert(len(images.shape) == 4)
            is_tensor = torch.is_tensor(images)
            if not is_tensor:
                images = torch.Tensor(images)
            latent = self._model.encode(images.to(self.device)).sample()
            norm_latent = latent / norm['std']
            encoded = torch.clamp(norm_latent, -1., 1.)
            
            if not is_tensor:
                return encoded.cpu().detach().numpy()
            else:
                return encoded

    # batch, channel, width, height
    def decode(self, norm_latent):
        with torch.no_grad():
            assert(len(norm_latent.shape) == 4)
            latent = norm_latent.to(self.device) * norm['std']
            return self._model.decode(latent)

#----------------------------------------------------------------------------
