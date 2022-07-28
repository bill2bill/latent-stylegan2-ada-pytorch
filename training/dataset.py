# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from enum import auto
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import shutil
import re
from torch_utils.misc import get_cache_dir

import datetime
import time

from transforms.latent import Autoencoder

import torchvision.transforms as transforms

from PIL import Image

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        # The images loaded are not encoded yet so different shape
        # assert list(image.shape) == self.image_shape
        # this check is valid for data in range of 0 - 255 but encoder converts to -1 to 1
        # assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                        # Path to directory or zip.
        resolution = None,           # Ensure specific resolution, None = highest available.
        ae = None,
        device = 'cuda:0',
        **super_kwargs              # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._ae = ae
        self._device = device

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]

        raw_image = self._load_raw_image(0)
        if ae:
            # raw_image = torch.HalfTensor(raw_image).to(self._device)
            raw_image = torch.FloatTensor(raw_image).to(self._device)
            
            raw_image = ae.encode(torch.unsqueeze(raw_image, 0))[0].cpu().detach().numpy()

        raw_shape = [len(self._image_fnames)] + list(raw_image.shape)

        # remove check if encoded as image size will be different
        if self._ae is None and resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        return image.transpose(2, 0, 1) # HWC => CHW

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

class ImageDataset(Dataset):
    def __init__(self, root='.', transform=None):
        self.image_paths = list(map(lambda path: f"{root}/{path}", os.listdir(root)))
        self.transform = transform
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.image_paths)

#----------------------------------------------------------------------------

class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self,
        path,                        # Path to directory or zip.
        resolution = None,           # Ensure specific resolution, None = highest available.
        batch_size = 32,
        workers = 2,
        ngpu = 4,
        rank = 0,
        max_size = None,
        clear = False, # Clear Cache
        cache = False, # Use data from cache
        **super_kwargs              # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._name = os.path.splitext(os.path.basename(self._path))[0]
        self._label_shape = None
        self._rank = rank
        self._ngpu = ngpu
        self._device = torch.device('cuda', rank)

        cache_dir = f"{get_cache_dir()}/latent_images"
        self._cache_dir = cache_dir
        if cache:
            self._length = max_size
            self._raw_shape = [self._length, 3, resolution, resolution]
        else:
            autoencoder = self._autoencoder()
            tsfm = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            dataset = ImageDataset(root=path, transform=tsfm)

            resolution = dataset[0].shape[1]

            fake_img = torch.randint(1, 255 + 1, (16, 3, resolution, resolution)).type(torch.FloatTensor).to(self._device)
            self._raw_shape = [len(dataset), *autoencoder.encode(fake_img).cpu().detach().numpy().shape[1:]]
            del fake_img

            dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers))
            self._length = len(dataloader)

            if clear:
                shutil.rmtree(cache_dir)

            stamp = datetime.datetime.now()

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                i = 0
                for elem in dataloader:
                    data = elem.type(torch.FloatTensor).to(self._device)
                    latent = autoencoder.encode(data).cpu().detach().numpy()
                    for z in latent:
                        cache_path = f'{cache_dir}/latent_{i}.npy'
                        i = i + 1
                        np.save(cache_path, z)
                    del data, latent
                print(f"Cache created in {round((datetime.datetime.now() - stamp).total_seconds() / 60, 0)} minutes")
            del autoencoder

        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        cache_path = f'{self._cache_dir}/latent_{idx}.npy'
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            assert isinstance(data, np.ndarray)
            
            # Use labels always false
            labels = self._get_raw_labels()

            return data, labels
        else:
            raise StopIteration

    def _get_raw_labels(self):
        return np.zeros([self._raw_shape[0], 0], dtype=np.float32)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def _autoencoder(self):
        return Autoencoder(self._device, ngpu = self._ngpu)  

    def decode(self, latent):
        autoencoder = self._autoencoder()
        latent = latent.to(self._device)
        img = autoencoder.decode(latent).cpu().detach()
        del latent, autoencoder
        return img

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------