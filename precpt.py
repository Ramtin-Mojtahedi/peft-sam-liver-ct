# Ramtin Mojtahedi
# perceptual_loss.py
import sys
import math
import os
import time
import pathlib
import warnings
import logging
import collections
from datetime import datetime
from typing import Union, Optional, List, Tuple, Text, BinaryIO
from collections import OrderedDict

import numpy as np
import PIL
import dateutil.tz
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models import vgg19
from torch.utils.data import DataLoader

from PIL import Image, ImageDraw, ImageFont, ImageColor
from tqdm import tqdm

from dataset import Dataset_FullImg, Dataset_DiscRegion
from lucent.optvis.param.spatial import pixel_image, fft_image, init_image
from lucent.optvis.param.color import to_valid_rgb

import cfg


# ─── Global definitions ──────────────────────────────────────────────────────
args = cfg.parse_args()
device = torch.device("cuda", args.gpu_device)

# VGG19 feature extractor (ImageNet-pretrained)
cnn = vgg19(pretrained=True).features.to(device).eval()

# Layers used for perceptual loss
content_layers_default = ["conv_4"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

# Normalization constants for ImageNet (VGG)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225], device=device)


# ------------------ Loss Modules ------------------ #
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        # Detach target so it is treated as constant (no gradient through target path)
        self.target = target.detach()
        self.loss = None

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized Gram matrix for style representation.

    x: [B, C, H, W]
    returns: [B*C, B*C] via flattening (classic NST formulation; normalized)
    """
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    """
    Normalize input images with ImageNet mean/std to match VGG training.
    """
    def __init__(self, mean, std):
        super().__init__()
        # Register as buffers so they move with .to(device) and are saved in state_dict
        mean_t = torch.as_tensor(mean).view(-1, 1, 1)
        std_t = torch.as_tensor(std).view(-1, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


# ------------------ Perceptual Loss Construction ------------------ #
def precpt_loss(
    cnn: nn.Module,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    content_layers: List[str] = content_layers_default,
    style_layers: List[str] = style_layers_default,
):
    """
    Build a Sequential(VGG + loss hooks) model that accumulates content/style losses.
    Returns (model, style_losses, content_losses).
    """
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses: List[ContentLoss] = []
    style_losses: List[StyleLoss] = []

    model = nn.Sequential(normalization)

    i = 0  # conv index
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)  # avoid in-place issues with inserted loss modules
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # Ensure style has 3 channels for VGG
            if style_img.size(1) == 1:
                style_img = style_img.expand(style_img.size(0), 3, style_img.size(2), style_img.size(3))
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim after the last inserted loss module
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            model = model[: (j + 1)]
            break

    return model, style_losses, content_losses


def run_precpt(
    cnn: nn.Module,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    input_img: torch.Tensor,
    style_weight: float = 1_000_000,
    content_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute a single scalar perceptual loss (style + content) for `input_img`
    against `style_img` and `content_img`.

    Note: this function overrides weights internally (kept as in your original code).
    """
    model, style_losses, content_losses = precpt_loss(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    # Optimize input only
    model.requires_grad_(False)
    input_img.requires_grad_(True)

    # Forward once to populate loss modules
    model(input_img)

    style_score = 0.0
    content_score = 0.0
    for sl in style_losses:
        style_score = style_score + sl.loss
    for cl in content_losses:
        content_score = content_score + cl.loss

    # Keep original behavior: override provided weights
    content_weight = 100
    style_weight = 100000

    style_score = style_score * style_weight
    content_score = content_score * content_weight

    loss = style_score + content_score
    return loss
