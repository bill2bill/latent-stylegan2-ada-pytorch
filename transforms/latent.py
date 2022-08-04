import os
import zipfile
import requests 

import numpy as np

import torch
import torch.nn as nn

# from ldm.models.autoencoder import AutoencoderKL
from ldm.models.autoencoder import VQModelInterface
from torch_utils.misc import get_cache_dir

# KL-f4
# DEFAULT_AE_CONFIG = {
#     "ddconfig": {
#         "double_z": True,
#         "z_channels": 3,
#         "resolution": 256,
#         "in_channels": 3,
#         "out_ch": 3,
#         "ch": 128,
#         "ch_mult": [1,2,4],
#         "num_res_blocks": 2,
#         "attn_resolutions": [],
#         "dropout": 0.0
#     },
#     "lossconfig": {
#         "target": "ldm.modules.losses.LPIPSWithDiscriminator",
#         "params": {
#             "disc_start": 50001,
#             "kl_weight": 1.0e-06,
#             "disc_weight": 0.5
#         }
#     },
#     "embed_dim": 3
# }
#VQ-f4
DEFAULT_AE_CONFIG = {
    "embed_dim": 3,
    "n_embed": 8192,
    "monitor": "val/rec_loss",
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
      "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
      "params": {
        "disc_conditional": False,
        "disc_in_channels": 3,
        "disc_start": 0,
        "disc_weight": 0.75,
        "codebook_weight": 1.0
      }
    }
}
CACHE_MODEL_DIR = 'pretrained_models'

#----------------------------------------------------------------------------
# GAN Normalising contanstants

norm = {
    "std": 70,
}

#----------------------------------------------------------------------------
# Converts images to a latent space. Then converts the latent space to a suitable range for GANs (-1 to 1)



def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_pre_trained_ae(url, folder):
    cache_dir = get_cache_dir()
    output_dir = f"{cache_dir}/{folder}"
    path = f"{output_dir}/model.ckpt"
    tmp_path = './tmp'
    if not os.path.exists(path):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        download_url(url, tmp_path)

        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def setup():
    # download_pre_trained_ae("https://ommer-lab.com/files/latent-diffusion/kl-f4.zip", CACHE_MODEL_DIR)
    download_pre_trained_ae("https://ommer-lab.com/files/latent-diffusion/vq-f4.zip", CACHE_MODEL_DIR)

class VQModelInterface2(VQModelInterface):
    def __init__(self, *args, **kwargs):
        self.ddconfig = kwargs['ddconfig']
        super().__init__(*args, **kwargs)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        #Hacky Fix
        self.quant_conv = torch.nn.Conv2d(2 * self.ddconfig["z_channels"], 2 * self.embed_dim, 1)
        super().init_from_ckpt(path, ignore_keys = ignore_keys)

class Autoencoder:
    def __init__(self, device, ngpu = None):
        self.device = device

        print(f'Creating Autoencoder on device: {device}')
        # model = AutoencoderKL(DEFAULT_AE_CONFIG["ddconfig"], DEFAULT_AE_CONFIG["lossconfig"], DEFAULT_AE_CONFIG["embed_dim"], ckpt_path=f"{get_cache_dir()}/{CACHE_MODEL_DIR}/model.ckpt")
        model = VQModelInterface2(embed_dim = DEFAULT_AE_CONFIG["embed_dim"], ddconfig = DEFAULT_AE_CONFIG["ddconfig"], lossconfig = DEFAULT_AE_CONFIG["lossconfig"], n_embed = DEFAULT_AE_CONFIG["n_embed"], ckpt_path=f"{get_cache_dir()}/{CACHE_MODEL_DIR}/model.ckpt")
        model = model.to(torch.device('cuda', 0))

        def parralel(model):
            if ngpu is None:
                model.requires_grad_(True)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], broadcast_buffers=False)
                model.requires_grad_(False)
                return model
            else:
                return nn.DataParallel(model, list(range(ngpu)))

        model.encoder = parralel(model.encoder)
        model.decoder = parralel(model.decoder)
        model.loss = parralel(model.loss)
        model.quant_conv = parralel(model.quant_conv)
        model.post_quant_conv = parralel(model.post_quant_conv)

        #for vq ae
        model.quantize = parralel(model.quantize)

        self._model = model

    # batch, channel, width, height
    def encode(self, images):
        with torch.no_grad():
            assert(len(images.shape) == 4)
            # encoded = self._model.encode(images).sample()
            encoded = self._model.encode(images)
            return encoded

    # batch, channel, width, height
    def decode(self, latent):
        with torch.no_grad():
            assert(len(latent.shape) == 4)
            latent = latent * norm['std']

            return self._model.decode(latent)

#----------------------------------------------------------------------------
