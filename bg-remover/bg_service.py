"""This file is a light refacto of the "inference.py" file from the original repo.
The refacto is necessary to adapt the service to a REST API.

As this is a light refacto, some functions are treated as black boxes.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torchvision import transforms
from PIL import Image

from src.models.modnet import MODNet


WEIGHTS_PATH = "pretrained/modnet_photographic_portrait_matting.ckpt"


class BGService:
    """Service for removing the background of an image.
    """
    def __init__(self):
        # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)
        if torch.cuda.is_available():
            modnet = modnet.cuda()
            weights = torch.load(WEIGHTS_PATH)
        else:
            weights = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
        modnet.load_state_dict(weights)
        modnet.eval()
        self.modnet = modnet

    def remove_background(self, img: Image.Image) -> Image:
        """ Removes the background of the image and replaces it with black (#000000).
        """
        tensor = self._image_to_tensor(img)
        tensor = self._resize_image_tensor(tensor)
        matte = self._infer_matte(tensor)
        foreground = self._apply_matte(img, matte)
        return foreground

    def _image_to_tensor(self, img: Image.Image) -> torch.Tensor:
        # unify image channels to 3
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]

        # convert image to PyTorch tensor
        tensor = Image.fromarray(img)
        # define image to tensor transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        tensor = transform(tensor)

        # add mini-batch dim
        tensor = tensor[None, :, :, :]

        return tensor

    def _resize_image_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """A black box"""
        ref_size = 512
        im_b, im_c, im_h, im_w = tensor.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        tensor = functional.interpolate(tensor, size=(im_rh, im_rw), mode='area')

        return tensor

    def _infer_matte(self, img: Image) -> torch.Tensor:
        _, _, matte = self.modnet(img.cuda() if torch.cuda.is_available() else img, True)
        return matte


    def _apply_matte(self, img: Image, matte) -> Image.Image:
        """ Replaces all the pixels that do not match the matte with black (#000000).
        """
        im_w, im_h = img.size
        matte = functional.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')

        img = np.asarray(img)
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]
        matte_img = np.repeat(np.asarray(matte_img)[:, :, None], 3, axis=2) / 255
        foreground = img * matte_img + np.full(img.shape, 0) * (1 - matte_img)

        return Image.fromarray(np.uint8(foreground))
