import os
import zipfile
import requests 

import numpy as np

import torch
import torch.nn as nn

from ldm.models.autoencoder import AutoencoderKL
from torch_utils.misc import get_cache_dir

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
CACHE_MODEL_DIR = 'pretrained_models'

#----------------------------------------------------------------------------
# GAN Normalising contanstants

norm = {
    "mean": 0,
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
    def __init__(self, device):
        self.device = device

        cache_dir = get_cache_dir()
        pl_sd = torch.load(f"{cache_dir}/{CACHE_MODEL_DIR}/model.ckpt")
        print(f'Creating Autoencoder on device: {device}')
        model = AutoencoderKL(DEFAULT_AE_CONFIG["ddconfig"], DEFAULT_AE_CONFIG["lossconfig"], DEFAULT_AE_CONFIG["embed_dim"])
        model.load_state_dict(pl_sd["state_dict"] ,strict=False)
        model = model.half()
        model.to(device)

        # modules = [model, mod el.quant_conv, model.post_quant_conv, model.encoder, model.decoder]
        modules = [model]

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
            tensor_device = 'cpu'
            if is_tensor:
                tensor_device = images.device
                images = images.type(torch.HalfTensor).to(tensor_device)
                # images = images.to(torch.float16).to(tensor_device)
            else:
                images = torch.HalfTensor(images)
            
            same_device = tensor_device == self.device

            if not same_device:
                images = images.to(self.device)

            encoded = self._model.encode(images).sample()
            encoded = encoded / norm['std']
            encoded = torch.clamp(encoded, -1., 1.)
            #convert to range 0 - 1
            encoded = (encoded + 1) / 2
            
            del images
            torch.cuda.empty_cache()

            if is_tensor:
                if same_device:
                    return encoded
                else:
                    return encoded.to(tensor_device)

            else:
                return encoded.cpu().detach().numpy()

    # batch, channel, width, height
    def decode(self, norm_latent):
        with torch.no_grad():
            assert(len(norm_latent.shape) == 4)
            tensor_device = norm_latent.device
            norm_latent = norm_latent.type(torch.HalfTensor).to(tensor_device)

            # norm_latent = (norm_latent - 1) * 2
            # latent = norm_latent.to(self.device) * norm['std']
            latent = norm_latent
            decoded = self._model.decode(latent.to(self.device))

            return decoded

#----------------------------------------------------------------------------
