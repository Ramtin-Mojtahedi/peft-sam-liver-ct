#!/usr/bin/env python3
"""
Universal utilities for MedSAM-Adapter fine-tuning
Cleaned & consolidated • 2025-08-08
Author: Ramtin Mojtahedi
"""

from __future__ import annotations

# ─── Standard library ─────────────────────────────────────────
import collections
import logging
import math
import os
import pathlib
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from inspect import signature
from typing import Any, BinaryIO, Dict, List, Optional, Text, Tuple, Union

# ─── Third-party – general ────────────────────────────────────
import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sns
from PIL import Image, ImageColor, ImageDraw, ImageFont
from tqdm import tqdm

# ─── Third-party – PyTorch / DL ───────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
from torch.autograd import Function, Variable
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision.models import vgg19
from collections.abc import Mapping, Sequence  # add this near your imports

# ─── Third-party – MONAI ──────────────────────────────────────
from monai.config import print_config
from monai.data import (
    CacheDataset,
    ThreadDataLoader,
    decollate_batch,
    load_decathlon_datalist,
    set_track_meta,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

from transformers import BitsAndBytesConfig

# ---- BitsAndBytes / Transformers shims --------------------------------------
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:
    class BitsAndBytesConfig:  # minimal shim
        def __init__(self, *args, **kwargs): pass

try:
    from bitsandbytes.nn import Linear4bit  # type: ignore
    HAS_BNB = True
except Exception:
    Linear4bit = None
    HAS_BNB = False

# PEFT utils (k-bit prep; convert shim if missing) ----------------------------
try:
    from peft.utils.other import prepare_model_for_kbit_training, convert_linear_layer  # type: ignore
except Exception:
    prepare_model_for_kbit_training = None  # not needed unless adapter_type == 'qlora'
    convert_linear_layer = None

# PEFT ------------------------------------------------------------------------
try:
    from peft import get_peft_model, LoraConfig  # type: ignore
except Exception:
    get_peft_model = None
    LoraConfig = None

# ----------------------------- Helpers ---------------------------------------

def has_bitsandbytes() -> bool:
    return HAS_BNB and (Linear4bit is not None)

def make_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16,
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_quant_type: str = "nf4",
) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit and has_bitsandbytes(),
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
    )

def prepare_kbit(model: nn.Module) -> nn.Module:
    try:
        sig = signature(prepare_model_for_kbit_training)
        kwargs = {}
        if "use_gradient_checkpointing" in sig.parameters:
            kwargs["use_gradient_checkpointing"] = False
        return prepare_model_for_kbit_training(model, **kwargs)
    except Exception as e:
        warnings.warn(f"prepare_model_for_kbit_training failed or unavailable: {e}")
        return model

# ----------------------------- Defaults & Targets ----------------------------
_LORA_DEFAULT    = dict(r=64,  lora_alpha=64,  lora_dropout=0.05)
_QLORA_DEFAULT   = dict(r=12,  lora_alpha=12,  lora_dropout=0.25)
_CONV_DEFAULT    = dict(r=128, lora_alpha=192, lora_dropout=0.05)  # ConvAdapter++
_ROSA_DEFAULT    = dict(r=256, lora_alpha=256, lora_dropout=0.04)  # RoSA v3 tuned
_DISCO_DEFAULT   = dict(r=64,  lora_alpha=64,  lora_dropout=0.05)

_TARGET_MODULES_REGEX = r"(?:^|\.)(qkv|proj|mlp\.(?:fc1|fc2|lin1|lin2)|fc1|fc2|patch_embed\.proj)$"
_TARGET_MODULES_LIST  = ["qkv", "proj", "fc1", "fc2", "patch_embed.proj"]

_ADAPTER_TAGS: Tuple[str, ...] = (
    "lora_A", "lora_B", ".lora_A.", ".lora_B.",
    ".A.weight", ".B.weight",
    ".conv1d.weight", ".conv1d.",
    ".conv.weight", ".DW", ".gate", ".ladder",
    ".gamma", ".beta", ".spec_gain", ".delta_s", ".rank_gate", ".adapter_gain",
    ".log_alpha", ".ctrl.", ".micro_ctrl.", ".shared_proj",
    ".sfA", ".sfB", ".sf_gate",
    ".pre_mix1d.", ".pre_mix2d.", ".pre_norm.", ".tok_norm.", ".out_norm.", ".dw.", ".mix.",
    ".mag_gain",
)

class AttrDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(); self.update(*args, **kwargs)
    def __getattr__(self, name):
        try: return self[name]
        except KeyError as e: raise AttributeError(name) from e
    def __setattr__(self, name, value):
        if name.startswith('_') or name in self.__dict__ or hasattr(AttrDict, name):
            return super().__setattr__(name, value)
        self[name] = self._wrap(value)
    def __delattr__(self, name):
        try: del self[name]
        except KeyError as e: raise AttributeError(name) from e
    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items(): super().__setitem__(k, self._wrap(v))
    @classmethod
    def _wrap(cls, v):
        if isinstance(v, Mapping): return AttrDict(v)
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            return type(v)(cls._wrap(x) for x in v)
        return v
    def to_dict(self):
        def unwrap(x):
            if isinstance(x, AttrDict): return {k: unwrap(v) for k, v in x.items()}
            if isinstance(x, list):     return [unwrap(i) for i in x]
            if isinstance(x, tuple):    return tuple(unwrap(i) for i in x)
            return x
        return unwrap(self)

def create_logger(save_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    logger   = logging.getLogger("MedSAM")
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if logger.handlers:
        for h in list(logger.handlers): logger.removeHandler(h)
    ch = logging.StreamHandler(stream=sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    if rank == 0:
        fh = logging.FileHandler(log_path, mode="a"); fh.setFormatter(fmt); logger.addHandler(fh)
    logger.propagate = False
    return logger

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad or not trainable_only))

def print_trainable_summary(model: nn.Module, prefix: str = "[params]") -> None:
    trn = count_parameters(model, True)
    tot = count_parameters(model, False)
    pct = (trn / max(1, tot)) * 100.0
    print(f"{prefix} {trn/1e6:.2f}M trainable / {tot/1e6:.2f}M total ({pct:.2f}%)")

def _resolve_device(gpu: Union[int, str, torch.device]) -> torch.device:
    if not torch.cuda.is_available(): return torch.device("cpu")
    if isinstance(gpu, torch.device): return gpu
    if isinstance(gpu, int):          return torch.device(f"cuda:{gpu}")
    if isinstance(gpu, str):
        g = gpu.lower()
        if g == "cpu":            return torch.device("cpu")
        if g.startswith("cuda"):  return torch.device(g)
        if g.isdigit():           return torch.device(f"cuda:{int(g)}")
    raise ValueError(f"Bad gpu_device: {gpu!r}")

def _flatten_tokens(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3: return t
    if t.dim() < 3:  raise ValueError(f"Need ≥3‑D, got {t.shape}")
    b, *spatial, r = t.shape
    return t.reshape(b, math.prod(spatial), r)

def _match_shape(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape == ref.shape: return src
    if src.ndim == ref.ndim == 3 and src.transpose(1, 2).shape == ref.shape:
        return src.transpose(1, 2)
    if {src.ndim, ref.ndim} == {3, 4} and src.shape[0] == ref.shape[0]:
        if src.ndim == 3:
            H, W = ref.shape[2], ref.shape[3]
            if H * W == src.shape[1]:
                return src.reshape(src.shape[0], H, W, -1).permute(0, 3, 1, 2)
        else:
            H, W = src.shape[2], src.shape[3]
            if H * W == ref.shape[1]:
                return src.permute(0, 2, 3, 1).reshape(src.shape[0], H*W, -1)
    if src.numel() // src.shape[0] == ref.numel() // ref.shape[0]:
        return src.reshape(ref.shape)
    raise RuntimeError(f"Cannot align {src.shape} → {ref.shape}")

# ----------------------------- LoRA base -------------------------------------

class _LoRALinearBase(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, p: float):
        super().__init__()
        self.base  = base
        self.r     = r
        self.scale = alpha / max(1, r)
        base.weight.requires_grad_(False)
        if base.bias is not None: base.bias.requires_grad_(False)
        self.A    = nn.Linear(base.in_features, r, bias=False)
        self.B    = nn.Linear(r, base.out_features, bias=False)
        self.drop = nn.Dropout(p)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

# ----------------------------- RoSA v3d --------------------------------------

class HardConcreteGate(nn.Module):
    def __init__(self, r: int, temperature: float = 2/3, gamma: float = -0.1, zeta: float = 1.1, p_keep_init: float = 0.95):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(r))
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self._l0_hook_handle = None
        self.reset_keep_prob(p_keep_init)
    def reset_keep_prob(self, p_keep: float):
        p = float(max(1e-5, min(1 - 1e-5, p_keep)))
        with torch.no_grad():
            self.log_alpha.copy_(torch.full_like(self.log_alpha, math.log(p) - math.log(1 - p)))
    def set_temperature(self, t: float): self.temperature = float(max(1e-3, t))
    def sample(self, training: bool) -> torch.Tensor:
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u) - torch.log(1-u) + self.log_alpha) / self.temperature)
        else:
            s = torch.sigmoid(self.log_alpha)
        s = s * (self.zeta - self.gamma) + self.gamma
        return s.clamp(0, 1)
    def l0(self) -> torch.Tensor:
        offset = - self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha + offset)
    def attach_l0_grad_hook(self, l0_lambda: float):
        if self._l0_hook_handle is not None:
            try: self._l0_hook_handle.remove()
            except Exception: pass
            self._l0_hook_handle = None
        if l0_lambda <= 0: return
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                offset = - self.temperature * math.log(-self.gamma / self.zeta)
                z = self.log_alpha.detach() + offset
                s = torch.sigmoid(z)
                add = float(l0_lambda) * s * (1.0 - s)
            return grad + add
        self._l0_hook_handle = self.log_alpha.register_hook(_hook)

class RoSAGateController(nn.Module):
    def __init__(self, in_dim: int, r: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=True)
        self.fc1  = nn.Linear(in_dim, hidden, bias=True)
        self.fc2  = nn.Linear(hidden, r, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:   pooled = x.mean(dim=(2, 3))
        elif x.dim() == 3: pooled = x.mean(dim=1)
        elif x.dim() == 2: pooled = x
        else:              pooled = x.view(x.size(0), -1)
        h = F.gelu(self.fc1(self.norm(pooled)))
        return torch.sigmoid(self.fc2(h))

def _softplus_gain(param: torch.Tensor) -> torch.Tensor:
    return F.softplus(param)

class RoSAAdapter_Linear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, p: float, topk_eval: int = 0,
                 *, ema_beta: float = 0.995, ema_boost: float = 0.35, energy_balance: bool = True,
                 safefloor_r: int = 8, safefloor_init: float = 0.12,
                 module_tag: str = "", k_scale: float = 1.0, rankdrop_p: float = 0.10,
                 score_temp: float = 1.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("RoSAAdapter_Linear expects nn.Linear as base.")
        self.base = base
        self.r = int(r)
        self.scale = alpha / max(1, r)
        self.topk_eval = int(topk_eval)
        self.train_topk = 0
        self.train_mix_lambda = 1.0
        self.eps = 1e-6
        self.module_tag = str(module_tag)
        self.k_scale_default = float(k_scale)
        self.rankdrop_p = float(rankdrop_p)
        # --- schedule bounds for teacher mix ---
        self.lam_min, self.lam_max = 0.05, 0.90
        self.score_temp = float(max(0.5, score_temp))

        # ---- RoSA‑OPF floor controls (train vs eval) ----
        # (Train floor ramps during warm‑up via schedule_rosa_from_args; eval floor is fixed.)
        self.fc2_floor_train_frac = 0.15
        self.fc2_floor_eval_frac  = 0.33

        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        # SVD of normalized base.weight to build spectral bases
        with torch.no_grad():
            W  = base.weight.detach().to(torch.float32)
            s0 = torch.linalg.norm(W, dim=1).clamp_min(self.eps)
            D  = W / s0[:, None]
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            k  = min(U.size(1), Vh.size(0))
            r_eff = min(self.r, k)
            U_r  = U[:, :r_eff].contiguous()
            S_r  = S[:r_eff].contiguous()
            V_rt = Vh[:r_eff, :].contiguous()
        self.r = r_eff

        dtype, device = base.weight.dtype, base.weight.device
        bf16_ok = (device.type == "cuda") and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        buf_dtype = (torch.bfloat16 if bf16_ok else dtype)
        self.register_buffer("U_r",  U_r.to(dtype=buf_dtype, device=device))
        self.register_buffer("S_r",  S_r.to(dtype=buf_dtype, device=device))
        self.register_buffer("V_rt", V_rt.to(dtype=buf_dtype, device=device))
        self.register_buffer("s0",   s0.to(dtype=buf_dtype, device=device))

        # Learnable spectral parameters / gates
        self.gamma     = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.beta      = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.rank_gate = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        # stronger initial spectral gain
        self.spec_gain = nn.Parameter(torch.tensor(math.log(math.expm1(0.8)), dtype=dtype, device=device))
        self.delta_s   = nn.Parameter(torch.zeros(base.out_features, dtype=dtype, device=device))
        self.drop      = nn.Dropout(p)

        # Instance gating & light mixers
        self.ctrl     = RoSAGateController(base.in_features, self.r, hidden=128)
        self.hard     = HardConcreteGate(self.r, p_keep_init=0.95)
        self.tok_norm = nn.LayerNorm(base.in_features, elementwise_affine=True)
        self.pre_norm = RMSNormLastDim(self.r)
        self.pre_mix1d= MSDWConv1d(self.r, ks=(3, 5), dilations=(1, 2))

        # EMA tracking for energy‑balance & score boosting
        self.ema_beta = float(ema_beta)
        self.ema_boost= float(ema_boost)
        self.energy_balance = bool(energy_balance)
        self.register_buffer("ema_score", torch.zeros(self.r, dtype=dtype, device=device))
        self.register_buffer("ema_dense_en", torch.tensor(1.0, dtype=dtype, device=device))
        self.register_buffer("ema_sparse_en", torch.tensor(1.0, dtype=dtype, device=device))

        # Optional Safe‑Floor low‑rank backup path
        self.safefloor_r = int(max(0, safefloor_r))
        if self.safefloor_r > 0:
            self.sfA = nn.Linear(base.in_features, self.safefloor_r, bias=False, device=device, dtype=dtype)
            self.sfB = nn.Linear(self.safefloor_r, base.out_features, bias=False, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.sfA.weight, a=math.sqrt(5))
            nn.init.zeros_(self.sfB.weight)
            self.sf_gate = nn.Parameter(torch.tensor(float(safefloor_init), dtype=dtype, device=device))
            self.register_buffer("sf_scale", torch.tensor(1.0, dtype=dtype, device=device))
        else:
            self.sfA = self.sfB = self.sf_gate = None
            self.register_buffer("sf_scale", torch.tensor(1.0, dtype=dtype, device=device))

        with torch.no_grad():
            nn.init.normal_(self.beta, mean=0.0, std=1e-2)

    # ---- runtime setters ----------------------------------------------------
    def set_eval_topk(self, k: int):           self.topk_eval = max(0, int(k))
    def set_train_topk(self, k: int):          self.train_topk = max(0, int(k))
    def set_train_mix(self, lam: float):       self.train_mix_lambda = float(max(0.0, min(1.0, lam)))
    def set_rankdrop(self, p: float):          self.rankdrop_p = float(max(0.0, min(1.0, p)))
    def set_sf_scale(self, s: float):          self.sf_scale.fill_(float(max(0.05, min(2.0, s))))

    # NEW: class‑level method (callable from scheduler/helper)
    def set_fc2_floor(self, train_frac=None, eval_frac=None):
        if train_frac is not None:
            self.fc2_floor_train_frac = float(max(0.0, min(1.0, train_frac)))
        if eval_frac is not None:
            self.fc2_floor_eval_frac  = float(max(0.0, min(1.0, eval_frac)))

    # ---- k-scaling with module‑aware policy + fc2 floor ---------------------
    def _scaled_k(self, k_raw: int) -> int:
        """
        Per‑module scaling for eval/train Top‑K.
        Keep attention / mlp.fc1 lean for compute; protect mlp.fc2 with a small floor.
        """
        if k_raw <= 0:
            return k_raw
        ks  = self.k_scale_default
        tag = (self.module_tag or "").lower()

        # Lean attention
        if ("attn" in tag) and ("qkv" in tag or "proj" in tag):
            ks *= 0.95
            return int(max(1, min(self.r, round(k_raw * ks))))

        # Lean fc1
        if ("mlp" in tag) and ("fc1" in tag or "lin1" in tag):
            ks *= 0.85
            return int(max(1, min(self.r, round(k_raw * ks))))

        # fc2 gets a floor (train vs eval)
        if ("mlp" in tag) and ("fc2" in tag or "lin2" in tag):
            ks *= 0.75
            scaled = int(round(k_raw * ks))
            floor_frac = (self.fc2_floor_eval_frac if not self.training else self.fc2_floor_train_frac)
            floor = max(1, int(round(floor_frac * self.r)))
            return int(min(self.r, max(floor, scaled)))

        # default
        return int(max(1, min(self.r, round(k_raw * ks))))

    # ---- gain/energy helpers -----------------------------------------------
    def _norm_preserving_gain(self, k_use: int) -> torch.Tensor:
        active = max(1, k_use) if k_use > 0 else self.r
        return self.scale * _softplus_gain(self.spec_gain) / math.sqrt(active)

    def _energy_correction(self, k_use: int) -> torch.Tensor:
        if not self.energy_balance or k_use <= 0:
            return torch.tensor(1.0, device=self.U_r.device, dtype=self.U_r.dtype)
        ratio = torch.sqrt((self.ema_dense_en + self.eps) / (self.ema_sparse_en + self.eps))
        return ratio.clamp(0.25, 4.0)

    # ---- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (+ DoRA-like magnitude on s0)
        y = self.base(x)
        s0 = self.s0.to(y.dtype) if self.s0.dtype != y.dtype else self.s0
        y = y * (1.0 + self.delta_s / (s0 + self.eps))

        # Tokens as (B,N,C) regardless of (B,C,H,W) vs (B,N,C)
        tokens = x.unsqueeze(1) if x.dim() == 2 else _flatten_tokens(x)
        B, N, C = tokens.shape

        # Instance gating uses normalized tokens
        tokens_n = self.tok_norm(tokens)

        # Spectral params / gates
        S_r = self.S_r.to(self.gamma.dtype) if self.S_r.dtype != self.gamma.dtype else self.S_r
        d      = (S_r * self.gamma + self.beta)          # (r,)
        g_base = torch.sigmoid(self.rank_gate)           # (r,)
        g_inst = self.ctrl(tokens_n)                     # (B,r)
        g_hc   = self.hard.sample(self.training)         # (r,)
        g      = g_inst * g_base.view(1,-1) * g_hc.view(1,-1)   # (B,r)

        # optional train-time rankdrop
        if self.training and self.rankdrop_p > 0.0:
            drop = (torch.rand_like(self.rank_gate) > self.rankdrop_p).to(g.dtype)
            g = g * drop.view(1, -1)

        # EMA score update (uses eval HC sample)
        with torch.no_grad():
            g_hc_mean = self.hard.sample(training=False)
            g_mean = g_inst * g_base.view(1,-1) * g_hc_mean.view(1,-1)
            s_mean = (d.view(1,-1) * g_mean).abs().mean(dim=0)   # (r,)
            self.ema_score.mul_(self.ema_beta).add_((1 - self.ema_beta) * s_mean)

        # Project tokens to spectral codes (B,N,r) and pre-mix in r-dim
        V_rt = self.V_rt.to(tokens_n.dtype) if self.V_rt.dtype != tokens_n.dtype else self.V_rt
        T_all = F.linear(tokens_n, V_rt)                         # (B,N,r)
        T_all = self.pre_norm(T_all)
        T_all = self.pre_mix1d(T_all.transpose(1, 2)).transpose(1, 2)

        # Choose eval/train Top‑K
        k_raw = self.train_topk if self.training else self.topk_eval
        k_use = self._scaled_k(k_raw)
        gain_sparse = self._norm_preserving_gain(k_use)

        if k_use <= 0 or k_use >= self.r:
            # Dense path
            t = self.drop(T_all) * (d.view(1,1,-1) * g.view(B,1,-1))    # (B,N,r)
            U_r = self.U_r.to(t.dtype) if self.U_r.dtype != t.dtype else self.U_r
            y_dense = F.linear(t, U_r)                                  # (B,N,out)
            y_dense = _match_shape(y_dense, y)
            out_delta = y_dense
            with torch.no_grad():
                self.ema_dense_en.mul_(self.ema_beta).add_((1 - self.ema_beta) * y_dense.pow(2).mean())
        else:
            # Sparse path: select k channels AFTER pre-mix
            scores_base = (d.view(1,-1) * g).abs()                      # (B,r)
            scores = scores_base + self.ema_boost * self.ema_score.view(1,-1)
            if self.score_temp != 1.0:
                scores = scores.clamp_min(1e-6).pow(self.score_temp)
            idx = torch.topk(scores, k=k_use, dim=1).indices            # (B,k)
            idx_expand = idx.unsqueeze(1).expand(B, N, k_use)           # (B,N,k)
            tb = torch.gather(T_all, dim=2, index=idx_expand)           # (B,N,k)

            # Gather corresponding U vectors and weights
            U_t   = self.U_r.t()
            U_t   = U_t.to(tb.dtype) if U_t.dtype != tb.dtype else U_t
            U_exp = U_t.unsqueeze(0).expand(B, -1, -1)                  # (B,out,r)
            U_sel_t = torch.gather(U_exp, 1, idx.unsqueeze(-1).expand(-1, -1, U_exp.size(2)))
            U_sel = U_sel_t.transpose(1, 2)                             # (B,out,k)
            w_sel = torch.gather(d.view(1,-1).expand(B,-1), 1, idx) * torch.gather(g, 1, idx)  # (B,k)

            tb = self.drop(tb) * w_sel.unsqueeze(1)                     # (B,N,k)
            y_sparse = torch.einsum('bnk,bok->bno', tb, U_sel)          # (B,N,out)
            y_sparse = _match_shape(y_sparse, y)

            if self.training:
                with torch.no_grad():
                    g_teacher = g_inst * g_base.view(1,-1) * self.hard.sample(training=False).view(1,-1)
                    t_teacher = T_all * (d.view(1,1,-1) * g_teacher.view(B,1,-1))  # (B,N,r)
                    U_r = self.U_r.to(t_teacher.dtype) if self.U_r.dtype != t_teacher.dtype else self.U_r
                    y_dense = F.linear(t_teacher, U_r)
                    y_dense = _match_shape(y_dense, y)
                    self.ema_dense_en.mul_(self.ema_beta).add_((1 - self.ema_beta) * y_dense.pow(2).mean())
                    self.ema_sparse_en.mul_(self.ema_beta).add_((1 - self.ema_beta) * y_sparse.pow(2).mean())
                lam = float(self.train_mix_lambda)
                corr = self._energy_correction(k_use)
                out_delta = lam * y_dense + (1.0 - lam) * (corr * y_sparse)
            else:
                out_delta = y_sparse

        # SafeFloor on base output
        if (self.safefloor_r > 0) and (self.sfA is not None):
            sf = self.sfB(self.sfA(x))
            y = y + (self.sf_scale * self.scale * F.softplus(self.sf_gate)) * sf

        return y + gain_sparse * out_delta

class RoSAAdapter_Conv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int, alpha: int, p: float, topk_eval: int = 0,
                 *, ema_beta: float = 0.995, ema_boost: float = 0.35, energy_balance: bool = True,
                 safefloor_r: int = 8, safefloor_init: float = 0.12,
                 module_tag: str = "", k_scale: float = 1.0, rankdrop_p: float = 0.10,
                 score_temp: float = 1.0):
        super().__init__()
        if not isinstance(base, nn.Conv2d) or getattr(base, "groups", 1) != 1:
            raise TypeError("RoSAAdapter_Conv2d expects standard nn.Conv2d (groups==1).")

        self.base = base
        self.r = int(r)
        self.scale = alpha / max(1, r)
        self.topk_eval = int(topk_eval)
        self.train_topk = 0
        self.train_mix_lambda = 1.0
        self.eps = 1e-6
        self.module_tag = str(module_tag)
        self.k_scale_default = float(k_scale)
        self.rankdrop_p = float(rankdrop_p)
        # slightly tighter mix band for conv path
        self.lam_min, self.lam_max = 0.20, 0.95
        self.score_temp = float(max(0.5, score_temp))

        # ---- RoSA‑OPF floor controls (train vs eval) ----
        self.fc2_floor_train_frac = 0.15
        self.fc2_floor_eval_frac  = 0.33

        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        self.stride, self.padding, self.dilation = base.stride, base.padding, base.dilation
        self.in_c, self.out_c = base.in_channels, base.out_channels
        kH, kW = base.kernel_size

        with torch.no_grad():
            W = base.weight.detach().reshape(self.out_c, -1).to(torch.float32)
            s0 = torch.linalg.norm(W, dim=1).clamp_min(1e-6)
            D  = W / s0[:, None]
            U, S, Vh = torch.linalg.svd(D, full_matrices=False)
            k = min(U.size(1), Vh.size(0))
            r_eff = min(self.r, k)
            U_r  = U[:, :r_eff].contiguous()
            S_r  = S[:r_eff].contiguous()
            V_rT = Vh[:r_eff, :].contiguous()
            Kv   = V_rT.view(r_eff, self.in_c, kH, kW).contiguous()
            Ku   = U_r.view(self.out_c, r_eff, 1, 1).contiguous()
        self.r = r_eff

        dtype, device = base.weight.dtype, base.weight.device
        bf16_ok = (device.type == "cuda") and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        buf_dtype = (torch.bfloat16 if bf16_ok else dtype)
        self.register_buffer("Kv",  Kv.to(dtype=buf_dtype, device=device))
        self.register_buffer("Ku",  Ku.to(dtype=buf_dtype, device=device))
        self.register_buffer("S_r", S_r.to(dtype=buf_dtype, device=device))
        self.register_buffer("s0",  s0.to(dtype=buf_dtype, device=device))

        self.gamma     = nn.Parameter(torch.zeros(self.r,     dtype=dtype, device=device))
        self.beta      = nn.Parameter(torch.zeros(self.r,     dtype=dtype, device=device))
        self.rank_gate = nn.Parameter(torch.zeros(self.r,     dtype=dtype, device=device))
        self.spec_gain = nn.Parameter(torch.tensor(math.log(math.expm1(0.8)), dtype=dtype, device=device))
        self.delta_s   = nn.Parameter(torch.zeros(self.out_c, dtype=dtype, device=device))
        self.drop2d    = nn.Dropout2d(p)

        self.out_norm = nn.GroupNorm(num_groups=1, num_channels=self.out_c, affine=True)
        self.ctrl     = RoSAGateController(self.out_c, self.r, hidden=128)
        self.hard     = HardConcreteGate(self.r, p_keep_init=0.95)

        self.ema_beta = float(ema_beta)
        self.ema_boost= float(ema_boost)
        self.energy_balance = bool(energy_balance)
        self.register_buffer("ema_score", torch.zeros(self.r, dtype=dtype, device=device))
        self.register_buffer("ema_dense_en", torch.tensor(1.0, dtype=dtype, device=device))
        self.register_buffer("ema_sparse_en", torch.tensor(1.0, dtype=dtype, device=device))

        self.pre_mix2d = MSDWConv2d(self.r, ks=(3, 3), dilations=(1, 2))

        # Optional Safe‑Floor 1x1 path on outputs
        self.safefloor_r = int(max(0, safefloor_r))
        if self.safefloor_r > 0:
            self.sfA = nn.Conv2d(self.out_c, self.safefloor_r, kernel_size=1, bias=False,
                                 device=device, dtype=dtype)
            self.sfB = nn.Conv2d(self.safefloor_r, self.out_c, kernel_size=1, bias=False,
                                 device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.sfA.weight, a=math.sqrt(5))
            nn.init.zeros_(self.sfB.weight)
            self.sf_gate = nn.Parameter(torch.tensor(float(safefloor_init), dtype=dtype, device=device))
            self.register_buffer("sf_scale", torch.tensor(1.0, dtype=dtype, device=device))
        else:
            self.sfA = self.sfB = self.sf_gate = None
            self.register_buffer("sf_scale", torch.tensor(1.0, dtype=dtype, device=device))

        with torch.no_grad():
            nn.init.normal_(self.beta, mean=0.0, std=1e-2)

    # ---- runtime setters ----------------------------------------------------
    def set_eval_topk(self, k: int):           self.topk_eval = max(0, int(k))
    def set_train_topk(self, k: int):          self.train_topk = max(0, int(k))
    def set_train_mix(self, lam: float):       self.train_mix_lambda = float(max(0.0, min(1.0, lam)))
    def set_rankdrop(self, p: float):          self.rankdrop_p = float(max(0.0, min(1.0, p)))
    def set_sf_scale(self, s: float):
        s = float(max(0.05, min(2.0, s))); self.sf_scale.fill_(s)

    # NEW: class‑level method
    def set_fc2_floor(self, train_frac=None, eval_frac=None):
        if train_frac is not None:
            self.fc2_floor_train_frac = float(max(0.0, min(1.0, train_frac)))
        if eval_frac is not None:
            self.fc2_floor_eval_frac  = float(max(0.0, min(1.0, eval_frac)))

    # ---- k-scaling with module‑aware policy + fc2 floor ---------------------
    def _scaled_k(self, k_raw: int) -> int:
        """
        Same policy for Conv2d spectral path: lean attn/fc1; fc2 gets a small floor.
        """
        if k_raw <= 0:
            return k_raw

        ks  = self.k_scale_default
        tag = (self.module_tag or "").lower()

        if ("attn" in tag) and ("qkv" in tag or "proj" in tag):
            ks *= 0.95
            return int(max(1, min(self.r, round(k_raw * ks))))

        if ("mlp" in tag) and ("fc1" in tag or "lin1" in tag):
            ks *= 0.85
            return int(max(1, min(self.r, round(k_raw * ks))))

        if ("mlp" in tag) and ("fc2" in tag or "lin2" in tag):
            ks *= 0.75
            scaled = int(round(k_raw * ks))
            floor_frac = (self.fc2_floor_eval_frac if not self.training else self.fc2_floor_train_frac)
            floor = max(1, int(round(floor_frac * self.r)))
            return int(min(self.r, max(floor, scaled)))

        return int(max(1, min(self.r, round(k_raw * ks))))

    # ---- gains/energy helpers ----------------------------------------------
    def _norm_preserving_gain(self, k_use: int) -> torch.Tensor:
        active = max(1, k_use) if k_use > 0 else self.r
        return self.scale * F.softplus(self.spec_gain) / math.sqrt(active)

    def _energy_correction(self, k_use: int) -> torch.Tensor:
        if not self.energy_balance or k_use <= 0:
            return torch.tensor(1.0, device=self.Kv.device, dtype=self.Kv.dtype)
        ratio = torch.sqrt((self.ema_dense_en + 1e-6) / (self.ema_sparse_en + 1e-6))
        return ratio.clamp(0.25, 4.0)

    # ---- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        s0 = self.s0.to(y.dtype) if self.s0.dtype != y.dtype else self.s0
        y = y * (1.0 + (self.delta_s / (s0 + 1e-6))).view(1, -1, 1, 1)

        B = y.size(0)
        y_n = self.out_norm(y)

        S_r = self.S_r.to(self.gamma.dtype) if self.S_r.dtype != self.gamma.dtype else self.S_r
        d      = (S_r * self.gamma + self.beta)
        g_base = torch.sigmoid(self.rank_gate)
        g_inst = self.ctrl(y_n)
        g_hc   = self.hard.sample(self.training)
        g_full = g_inst * g_base.view(1, -1) * g_hc.view(1, -1)

        if self.training and self.rankdrop_p > 0.0:
            drop = (torch.rand_like(self.rank_gate) > self.rankdrop_p).to(g_full.dtype)
            g_full = g_full * drop.view(1, -1)

        with torch.no_grad():
            g_hc_mean = self.hard.sample(training=False)
            g_mean = g_inst * g_base.view(1, -1) * g_hc_mean.view(1, -1)
            s_mean = (d.view(1, -1) * g_mean).abs().mean(dim=0)
            self.ema_score.mul_(self.ema_beta).add_((1 - self.ema_beta) * s_mean)

        k_raw = self.train_topk if self.training else self.topk_eval
        k_use = self._scaled_k(k_raw)
        gain_sparse = self._norm_preserving_gain(k_use)

        Kv = self.Kv.to(x.dtype) if self.Kv.dtype != x.dtype else self.Kv
        # pre-mix while channels == r (before Top‑K)
        T_all = F.conv2d(x, Kv, bias=None, stride=self.stride, padding=self.padding,
                         dilation=self.dilation, groups=1)
        T_all = self.pre_mix2d(T_all)

        if k_use <= 0 or k_use >= self.r:
            # Dense path
            t = self.drop2d(T_all) * (d.view(1, -1, 1, 1) * g_full.view(B, -1, 1, 1))
            Ku = self.Ku.to(t.dtype) if self.Ku.dtype != t.dtype else self.Ku
            y_dense = F.conv2d(t, Ku, bias=None, stride=1, padding=0, dilation=1, groups=1)
            out_delta = y_dense
            with torch.no_grad():
                self.ema_dense_en.mul_(self.ema_beta).add_((1 - self.ema_beta) * y_dense.pow(2).mean())
        else:
            # Sparse path: choose k per‑sample after pre‑mix
            scores_base = (d.view(1, -1) * g_full).abs()
            scores = scores_base + self.ema_boost * self.ema_score.view(1, -1)
            if self.score_temp != 1.0:
                scores = scores.clamp_min(1e-6).pow(self.score_temp)

            y_list = []
            U = self.Ku
            for b in range(B):
                idx = torch.topk(scores[b], k=k_use, largest=True).indices
                Tb_sel = T_all[b:b+1].index_select(dim=1, index=idx)   # (1, k_use, H', W')
                d_sel  = d.index_select(0, idx).view(1, -1, 1, 1)
                gi_sel = g_full[b].index_select(0, idx).view(1, -1, 1, 1)
                Tb_sel = self.drop2d(Tb_sel) * (d_sel * gi_sel)
                Ku_sel = U.index_select(1, idx)
                Ku_sel = Ku_sel.to(Tb_sel.dtype) if Ku_sel.dtype != Tb_sel.dtype else Ku_sel
                yb = F.conv2d(Tb_sel, Ku_sel, bias=None, stride=1, padding=0, dilation=1, groups=1)
                y_list.append(yb)
            y_sparse = torch.cat(y_list, dim=0)

            if self.training:
                with torch.no_grad():
                    g_teacher = g_inst * g_base.view(1, -1) * self.hard.sample(training=False).view(1, -1)
                    td = self.drop2d(T_all) * (d.view(1, -1, 1, 1) * g_teacher.view(B, -1, 1, 1))
                    Ku = self.Ku.to(td.dtype) if self.Ku.dtype != td.dtype else self.Ku
                    y_dense = F.conv2d(td, Ku, bias=None, stride=1, padding=0, dilation=1, groups=1)
                    self.ema_dense_en.mul_(self.ema_beta).add_((1 - self.ema_beta) * y_dense.pow(2).mean())
                    self.ema_sparse_en.mul_(self.ema_beta).add_((1 - self.ema_beta) * y_sparse.pow(2).mean())
                lam = float(self.train_mix_lambda)
                corr = self._energy_correction(k_use)
                out_delta = lam * y_dense + (1.0 - lam) * (corr * y_sparse)
            else:
                out_delta = y_sparse

        if (self.safefloor_r > 0) and (self.sfA is not None):
            sf = self.sfB(self.sfA(y))
            y = y + (self.sf_scale * self.scale * F.softplus(self.sf_gate)) * sf

        return y + gain_sparse * out_delta

# ----------------------------- Lightweight norms & mixers --------------------

class RMSNormLastDim(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

class RMSNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, c, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight

class MSDWConv1d(nn.Module):
    def __init__(self, r: int, ks: Tuple[int, int] = (3, 5), dilations: Tuple[int, int] = (1, 2)):
        super().__init__()
        assert len(ks) == len(dilations) and len(ks) >= 1
        self.dw = nn.ModuleList([
            nn.Conv1d(r, r, kernel_size=k, padding=(k // 2) * d, dilation=d, groups=r, bias=False)
            for k, d in zip(ks, dilations)
        ])
        self.mix = nn.Conv1d(r, r, kernel_size=1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = None
        for op in self.dw:
            y = op(x) if y is None else (y + op(x))
        y = y / math.sqrt(len(self.dw))
        y = F.gelu(y, approximate="tanh")
        return self.mix(y)

class MSDWConv2d(nn.Module):
    def __init__(self, r: int, ks: Tuple[int, int] = (3, 3), dilations: Tuple[int, int] = (1, 2)):
        super().__init__()
        assert len(ks) == len(dilations) and len(ks) >= 1
        self.dw = nn.ModuleList([
            nn.Conv2d(r, r, kernel_size=k, padding=(k // 2) * d, dilation=d, groups=r, bias=False)
            for k, d in zip(ks, dilations)
        ])
        self.mix = nn.Conv2d(r, r, kernel_size=1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = None
        for op in self.dw:
            y = op(x) if y is None else (y + op(x))
        y = y / math.sqrt(len(self.dw))
        y = F.gelu(y, approximate="tanh")
        return self.mix(y)

# ----------------------------- Conv‑Adapter++ --------------------------------

class ConvLoRA_Linear(_LoRALinearBase):
    def __init__(self, base: nn.Linear, r: int, alpha: int, p: float,
                 k: int = 3, share_conv1d: Optional[nn.Module] = None):
        super().__init__(base, r, alpha, p)
        if share_conv1d is None:
            ks = (k, max(k + 2, 5)) if isinstance(k, int) else (3, 5)
            self.conv1d = MSDWConv1d(r, ks=ks, dilations=(1, 2))
        else:
            self.conv1d = share_conv1d
        self.norm = RMSNormLastDim(r)
        with torch.no_grad():
            W = base.weight.detach().to(torch.float32)
            s0 = torch.linalg.norm(W, dim=1).clamp_min(1e-6)
        dtype, device = base.weight.dtype, base.weight.device
        self.register_buffer("s0", s0.to(dtype=dtype, device=device))
        self.delta_s       = nn.Parameter(torch.zeros(base.out_features, dtype=dtype, device=device))
        self.rank_gate     = nn.Parameter(torch.zeros(r, dtype=dtype, device=device))
        self.adapter_gain  = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        self.ctrl = RoSAGateController(base.in_features, r, hidden=128)
        with torch.no_grad():
            self.adapter_gain.fill_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        s0 = self.s0.to(out.dtype) if self.s0.dtype != out.dtype else self.s0
        out = out * (1.0 + self.delta_s / (s0 + 1e-6))
        tokens = x.unsqueeze(1) if x.dim() == 2 else _flatten_tokens(x)
        t = self.drop(self.A(x))
        t = t.unsqueeze(1) if t.dim() == 2 else _flatten_tokens(t)
        t = self.norm(t)
        t = t.transpose(1, 2)
        t = self.conv1d(t)
        t = t.transpose(1, 2)
        g_inst  = self.ctrl(tokens)
        g_stat  = torch.sigmoid(self.rank_gate)
        t = t * (g_stat.view(1, 1, -1) * g_inst.view(t.size(0), 1, -1))
        t = self.scale * torch.tanh(self.adapter_gain) * self.B(t)
        return out + _match_shape(t, out)

class ConvAdapter_Conv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int, alpha: int, p: float,
                 k: int = 3, share_dw: Optional[nn.Module] = None,
                 safefloor_r: int = 8, safefloor_init: float = 0.05):
        super().__init__()
        if getattr(base, "groups", 1) != 1:
            raise TypeError("ConvAdapter_Conv2d expects standard nn.Conv2d (groups==1).")

        self.base  = base
        self.r     = r
        self.scale = alpha / max(1, r)

        base.weight.requires_grad_(False)
        if base.bias is not None: base.bias.requires_grad_(False)

        with torch.no_grad():
            W  = base.weight.detach().reshape(base.out_channels, -1).to(torch.float32)
            s0 = torch.linalg.norm(W, dim=1).clamp_min(1e-6)
        dtype, device = base.weight.dtype, base.weight.device
        self.register_buffer("s0", s0.to(dtype=dtype, device=device))
        self.delta_s       = nn.Parameter(torch.zeros(base.out_channels, dtype=dtype, device=device))
        self.rank_gate     = nn.Parameter(torch.zeros(r, dtype=dtype, device=device))
        self.adapter_gain  = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

        self.out_norm = nn.GroupNorm(num_groups=1, num_channels=base.out_channels, affine=True)

        self.A    = nn.Conv2d(base.out_channels, r, kernel_size=1, bias=False)
        self.norm = RMSNorm2d(r)
        if share_dw is None:
            ks2 = (k, 3) if k != 3 else (3, 3)
            self.DW = MSDWConv2d(r, ks=ks2, dilations=(1, 2))
        else:
            self.DW = share_dw
        self.B    = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(p)
        self.ctrl = RoSAGateController(base.out_channels, r, hidden=128)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        with torch.no_grad():
            self.adapter_gain.fill_(0.1)

        self.safefloor_r = int(max(0, safefloor_r))
        if self.safefloor_r > 0:
            self.sfA = nn.Conv2d(base.out_channels, self.safefloor_r, kernel_size=1, bias=False,
                                 device=device, dtype=dtype)
            self.sfB = nn.Conv2d(self.safefloor_r, base.out_channels, kernel_size=1, bias=False,
                                 device=device, dtype=dtype)
            nn.init.kaiming_uniform_(self.sfA.weight, a=math.sqrt(5))
            nn.init.zeros_(self.sfB.weight)
            self.sf_gate = nn.Parameter(torch.tensor(float(safefloor_init), dtype=dtype, device=device))
        else:
            self.sfA = self.sfB = self.sf_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        s0 = self.s0.to(y.dtype) if self.s0.dtype != y.dtype else self.s0
        y = y * (1.0 + (self.delta_s / (s0 + 1e-6))).view(1, -1, 1, 1)
        y_n = self.out_norm(y)
        a = self.A(y_n); a = self.norm(a); a = self.drop(a); a = self.DW(a)
        g_inst = self.ctrl(y_n).view(y.size(0), -1, 1, 1)
        g_stat = torch.sigmoid(self.rank_gate).view(1, -1, 1, 1)
        a = a * (g_inst * g_stat)
        a = self.scale * torch.tanh(self.adapter_gain) * self.B(a)
        if (self.safefloor_r > 0) and (self.sfA is not None):
            sf = self.sfB(self.sfA(y))
            y = y + (self.scale * F.softplus(self.sf_gate)) * sf
        return y + a

# ----------------------------- DiSCo (Directional Spectral) ------------------

class DiSCoAdapter_Linear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, p: float, x_topk: int = 0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("DiSCoAdapter_Linear expects nn.Linear as base.")
        self.base = base
        self.r = int(r)
        self.scale = alpha / max(1, r)
        base.weight.requires_grad_(False)
        if base.bias is not None: base.bias.requires_grad_(False)

        with torch.no_grad():
            W = base.weight.detach().to(torch.float32)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = min(U.size(1), Vh.size(0))
            r_eff = min(self.r, k)
            U_r  = U[:, :r_eff].contiguous()
            S_r  = S[:r_eff].contiguous()
            V_rt = Vh[:r_eff, :].contiguous()

        self.r = r_eff
        dtype = base.weight.dtype; device = base.weight.device
        self.register_buffer("U_r",  U_r.to(dtype=dtype, device=device))
        self.register_buffer("S_r",  S_r.to(dtype=dtype, device=device))
        self.register_buffer("V_rt", V_rt.to(dtype=dtype, device=device))
        self.register_buffer("x_topk", torch.tensor(int(x_topk), dtype=torch.int64, device=device))

        out = base.out_features
        self.delta_s   = nn.Parameter(torch.zeros(out, dtype=dtype, device=device))
        self.gamma     = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.beta      = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.rank_gate = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.spec_gain = nn.Parameter(torch.zeros(1,   dtype=dtype, device=device))
        self.mag_gain  = nn.Parameter(torch.zeros(1,   dtype=dtype, device=device))
        self.drop      = nn.Dropout(p)

    def set_topk(self, k: int):
        self.x_topk.data = torch.tensor(int(max(0, k)), dtype=torch.int64, device=self.x_topk.device)

    def _g_mask(self) -> torch.Tensor:
        g = torch.sigmoid(self.rank_gate)
        k = int(self.x_topk.item())
        if k <= 0 or k >= g.numel(): return g
        _, idx = torch.topk(g, k, largest=True, sorted=False)
        mask = torch.zeros_like(g); mask[idx] = 1.0
        return g * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.base(x)
        mag = (1.0 + torch.tanh(self.delta_s))
        y_mag = y_base * mag

        t = F.linear(x, self.V_rt)
        t = self.drop(t)
        d = self.S_r * self.gamma + self.beta
        g = self._g_mask()
        t = t * (d * g)
        y_dir = F.linear(t, self.U_r)
        return y_mag + self.scale * torch.tanh(self.spec_gain) * y_dir

class DiSCoAdapter_Conv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r: int, alpha: int, p: float, x_topk: int = 0):
        super().__init__()
        if not isinstance(base, nn.Conv2d) or base.groups != 1:
            raise ValueError("DiSCoAdapter_Conv2d expects nn.Conv2d with groups==1.")
        self.base  = base
        self.r     = int(r)
        self.scale = alpha / max(1, r)
        base.weight.requires_grad_(False)
        if base.bias is not None: base.bias.requires_grad_(False)

        self.stride   = base.stride
        self.padding  = base.padding
        self.dilation = base.dilation
        self.in_c     = base.in_channels
        self.out_c    = base.out_channels
        kH, kW        = base.kernel_size

        with torch.no_grad():
            W = base.weight.detach().reshape(self.out_c, -1).to(torch.float32)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = min(U.size(1), Vh.size(0))
            r_eff = min(self.r, k)
            U_r  = U[:, :r_eff].contiguous()
            S_r  = S[:r_eff].contiguous()
            V_rT = Vh[:r_eff, :].contiguous()
            Kv   = V_rT.view(r_eff, self.in_c, kH, kW).contiguous()
            Ku   = U_r.view(self.out_c, r_eff, 1, 1).contiguous()

        self.r = r_eff
        dtype  = base.weight.dtype; device = base.weight.device
        self.register_buffer("Kv", Kv.to(dtype=dtype, device=device))
        self.register_buffer("Ku", Ku.to(dtype=dtype, device=device))
        self.register_buffer("S_r", S_r.to(dtype=dtype, device=device))
        self.register_buffer("x_topk", torch.tensor(int(x_topk), dtype=torch.int64, device=device))

        self.delta_s   = nn.Parameter(torch.zeros(self.out_c, dtype=dtype, device=device))
        self.gamma     = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.beta      = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.rank_gate = nn.Parameter(torch.zeros(self.r, dtype=dtype, device=device))
        self.spec_gain = nn.Parameter(torch.zeros(1,     dtype=dtype, device=device))
        self.mag_gain  = nn.Parameter(torch.zeros(1,     dtype=dtype, device=device))
        self.drop2d    = nn.Dropout2d(p)

    def set_topk(self, k: int):
        self.x_topk.data = torch.tensor(int(max(0, k)), dtype=torch.int64, device=self.x_topk.device)

    def _g_mask(self) -> torch.Tensor:
        g = torch.sigmoid(self.rank_gate)
        k = int(self.x_topk.item())
        if k <= 0 or k >= g.numel(): return g
        _, idx = torch.topk(g, k, largest=True, sorted=False)
        mask = torch.zeros_like(g); mask[idx] = 1.0
        return g * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_base = self.base(x)
        mag = (1.0 + torch.tanh(self.delta_s)).view(1, -1, 1, 1)
        y_mag = y_base * mag

        t = F.conv2d(x, self.Kv, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        t = self.drop2d(t)
        d = (self.S_r * self.gamma + self.beta).view(1, -1, 1, 1)
        g = self._g_mask().view(1, -1, 1, 1)
        t = t * (d * g)
        y_dir = F.conv2d(t, self.Ku, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return y_mag + self.scale * torch.tanh(self.spec_gain) * y_dir

def set_disco_topk(root: nn.Module, k: int) -> None:
    for m in root.modules():
        if hasattr(m, "set_topk") and callable(m.set_topk):
            m.set_topk(k)

# ----------------------------- Freeze & Target parsing -----------------------

def _freeze_backbone(model: nn.Module, verbose: bool = True) -> None:
    for p in model.parameters(): p.requires_grad_((False))
    for n, p in model.named_parameters():
        if any(tag in n for tag in _ADAPTER_TAGS):
            p.requires_grad_(True)
    if verbose:
        print_trainable_summary(model, "[freeze]")

def _parse_targets(target_modules: Optional[Union[str, List[str]]]):
    tm_list: Optional[List[str]] = None
    patt  : Optional[re.Pattern] = None
    if isinstance(target_modules, str) and not target_modules.strip().startswith("("):
        tm_list = [t.strip() for t in target_modules.split(",") if t.strip()]
    else:
        patt = re.compile(target_modules or _TARGET_MODULES_REGEX)
    return tm_list, patt

# ----------------------------- attach_adapter --------------------------------

def attach_adapter(
    model: nn.Module,
    args_or_type: Union[str, Any],
    *,
    steps: Optional[int] = None,
    num_virtual_tokens: int = 20,
    target_modules: Optional[Union[str, List[str]]] = None,
    share_st: bool = True,
    conv_kernel: int = 3,
) -> nn.Module:
    if isinstance(args_or_type, str):
        a = None
        atype = args_or_type.lower()
        rank  = None
    else:
        a = args_or_type
        atype = a.adapter_type.lower()
        target_modules = target_modules if target_modules is not None else getattr(a, "target_modules", None)
        steps = steps or getattr(a, "total_step", None)
        num_virtual_tokens = getattr(a, "num_virtual_tokens", num_virtual_tokens)
        share_st = getattr(a, "share_st", share_st)
        conv_kernel = getattr(a, "conv_kernel", conv_kernel)
        rank = getattr(a, "mid_dim", None)

    if atype in {"stadapter", "specadapter", "diso"}:
        atype = "disco"

    def _cfg_for(kind: str):
        if rank is not None:
            if kind == "qlora":
                return dict(r=rank, lora_alpha=rank, lora_dropout=_QLORA_DEFAULT["lora_dropout"])
            elif kind == "rosa":
                alpha_mul = float(os.environ.get("ROSA_ALPHA_MUL", "2.5"))
                drop = getattr(a, "rosa_dropout", _ROSA_DEFAULT["lora_dropout"]) if a is not None else _ROSA_DEFAULT["lora_dropout"]
                return dict(r=rank, lora_alpha=int(alpha_mul * rank), lora_dropout=drop)
            elif kind == "conv":
                return dict(r=rank, lora_alpha=max(rank, 2*rank), lora_dropout=_CONV_DEFAULT["lora_dropout"])
            else:
                return dict(r=rank, lora_alpha=rank, lora_dropout=_LORA_DEFAULT["lora_dropout"])
        if kind == "qlora": return _QLORA_DEFAULT
        if kind == "rosa":  return _ROSA_DEFAULT
        if kind == "conv":  return _CONV_DEFAULT
        return _LORA_DEFAULT

    tm_list, patt = _parse_targets(target_modules)

    # FULL
    if atype == "full":
        print("[attach_adapter] Using FULL fine‑tuning (no adapter).")
        for p in model.parameters(): p.requires_grad_(True)
        print_trainable_summary(model)
        return model

    # LoRA / QLoRA
    if atype in {"lora", "qlora"}:
        if get_peft_model is None or LoraConfig is None:
            raise RuntimeError("PEFT is required for LoRA/QLoRA but is not available in this environment.")
        cfg = _cfg_for(atype)
        lcfg = LoraConfig(
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            target_modules=tm_list or _TARGET_MODULES_LIST,
            task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, lcfg)
        _freeze_backbone(model)
        print_trainable_summary(model, f"[adapter:{atype}]")
        return model

    # RoSA
    if atype in {"rosa", "routed_spectral", "rosa_ic"}:
        cfg = _cfg_for("rosa")
        stats = {"linear": 0, "conv": 0}
        topk_eval = int(getattr(a, "x_topk", 0)) if a is not None else 0  # default dense eval

        def _get(name, default):
            return getattr(a, name, default) if a is not None else default

        rosa_kwargs = dict(
            topk_eval=topk_eval,
            ema_beta=float(_get("rosa_ema_beta", 0.995)),
            ema_boost=float(_get("rosa_ema_boost", 0.35)),
            energy_balance=bool(_get("rosa_energy_balance", True)),
            safefloor_r=int(_get("rosa_safefloor_r", 8)),
            safefloor_init=float(_get("rosa_safefloor_init", 0.08)),
            score_temp=float(_get("rosa_score_temp", 1.0)),
        )

        def inject(module: nn.Module, prefix: str = ""):
            for name, child in list(module.named_children()):
                full = f"{prefix}{name}" if prefix else name
                hit = bool(patt.search(full)) if patt else (name in (tm_list or []) or full.split(".")[-1] in (tm_list or []))
                if hit and isinstance(child, nn.Linear):
                    if (child.in_features < 64) or (child.out_features < 64):
                        inject(child, prefix=full + "."); continue
                    setattr(module, name, RoSAAdapter_Linear(
                        child, cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"],
                        module_tag=full, k_scale=float(_get("rosa_k_scale", 1.05) or 1.05),
                        rankdrop_p=float(_get("rosa_rankdrop_init", 0.10)),
                        **rosa_kwargs
                    ))
                    stats["linear"] += 1
                elif hit and isinstance(child, nn.Conv2d):
                    if getattr(child, "groups", 1) != 1:
                        inject(child, prefix=full + "."); continue
                    setattr(module, name, RoSAAdapter_Conv2d(
                        child, cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"],
                        module_tag=full, k_scale=float(_get("rosa_k_scale", 1.05) or 1.05),
                        rankdrop_p=float(_get("rosa_rankdrop_init", 0.10)),
                        **rosa_kwargs
                    ))
                    stats["conv"] += 1
                else:
                    inject(child, prefix=full + ".")

        inject(model)
        _freeze_backbone(model)

        enabled = 0
        for m in model.modules():
            if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
                for pname, p in m.named_parameters(recurse=False):
                    if pname in ("gamma", "beta", "spec_gain", "delta_s", "rank_gate"):
                        p.requires_grad_(True); enabled += p.numel()
                for p in m.ctrl.parameters():
                    p.requires_grad_(True); enabled += p.numel()
                m.hard.log_alpha.requires_grad_(True); enabled += m.hard.log_alpha.numel()
                if getattr(m, "safefloor_r", 0) > 0:
                    for p in getattr(m, "sfA", nn.Identity()).parameters(): p.requires_grad_(True); enabled += p.numel()
                    for p in getattr(m, "sfB", nn.Identity()).parameters(): p.requires_grad_(True); enabled += p.numel()
                    if hasattr(m, "sf_gate"): m.sf_gate.requires_grad_(True); enabled += m.sf_gate.numel()

        print(f"[adapter:rosa] attached: {stats['linear']} Linear, {stats['conv']} Conv2d")
        print(f"[adapter:rosa] trainable params: {enabled} ({enabled/1e6:.6f}M)")
        print_trainable_summary(model, "[adapter:rosa]")

        # NEW: allow CLI to control initial hard-concrete keep prob (less "sticky" early gates)
        try:
            p_keep_init = float(getattr(a, "rosa_p_keep_init", 0.95))
        except Exception:
            p_keep_init = 0.95
        for m in model.modules():
            if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
                try:
                    m.hard.reset_keep_prob(p_keep_init)
                except Exception:
                    pass

        l0_lambda = float(getattr(a, "rosa_l0", 0.0) or 0.0)
        if l0_lambda > 0:
            for m in model.modules():
                if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
                    m.hard.attach_l0_grad_hook(l0_lambda)

        if topk_eval and topk_eval > 0:
            for m in model.modules():
                if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
                    m.set_eval_topk(int(topk_eval))

        return model

    # DiSCo
    if atype == "disco":
        def _cfg_for_disco():
            if rank is not None:
                return dict(r=rank, lora_alpha=rank, lora_dropout=_DISCO_DEFAULT["lora_dropout"])
            return _DISCO_DEFAULT

        cfg = _cfg_for_disco()
        x_topk = int(getattr(args_or_type, "x_topk", 0)) if not isinstance(args_or_type, str) else 0

        stats = {"linear": 0, "conv": 0}
        def inject(module: nn.Module, prefix: str = ""):
            for name, child in list(module.named_children()):
                full = f"{prefix}{name}" if prefix else name
                hit = bool(patt.search(full)) if patt else (name in (tm_list or []) or full.split(".")[-1] in (tm_list or []))
                if hit and isinstance(child, nn.Linear):
                    if (child.in_features < 64) or (child.out_features < 64):
                        inject(child, prefix=full + "."); continue
                    setattr(module, name, DiSCoAdapter_Linear(child, cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"], x_topk=x_topk))
                    stats["linear"] += 1

                elif hit and isinstance(child, nn.Conv2d):
                    if getattr(child, "groups", 1) != 1:
                        inject(child, prefix=full + "."); continue
                    try:
                        spec = DiSCoAdapter_Conv2d(child, cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"], x_topk=x_topk)
                        setattr(module, name, spec)
                        stats["conv"] += 1
                    except Exception:
                        inject(child, prefix=full + "."); continue
                else:
                    inject(child, prefix=full + ".")

        inject(model)
        _freeze_backbone(model, verbose=False)
        enabled = 0
        for m in model.modules():
            if isinstance(m, (DiSCoAdapter_Linear, DiSCoAdapter_Conv2d)):
                for pname, p in m.named_parameters(recurse=False):
                    if pname in ("delta_s", "gamma", "beta", "rank_gate", "spec_gain", "mag_gain"):
                        p.requires_grad_(True); enabled += p.numel()
        print(f"[adapter:disco] attached: {stats['linear']} Linear, {stats['conv']} Conv2d")
        print(f"[adapter:disco] trainable params: {enabled} ({enabled/1e6:.6f}M)")
        print_trainable_summary(model, "[adapter:disco]")
        return model

    # Conv‑Adapter++
    if atype == "conv":
        cfg = _cfg_for("conv")
        k = int(conv_kernel) if isinstance(conv_kernel, int) else 3
        shared_c1d = MSDWConv1d(cfg["r"], ks=(k, max(k + 2, 5)), dilations=(1, 2))
        ks2 = (k, 3) if k != 3 else (3, 3)
        shared_dw2 = MSDWConv2d(cfg["r"], ks=ks2, dilations=(1, 2))

        conv_sf_r    = int(getattr(a, "conv_safefloor_r", 8)) if a is not None else 8
        conv_sf_init = float(getattr(a, "conv_safefloor_init", 0.05)) if a is not None else 0.05

        def inject(module: nn.Module, prefix: str = ""):
            for name, child in list(module.named_children()):
                full = f"{prefix}{name}" if prefix else name
                hit  = bool(patt.search(full)) if patt else (name in (tm_list or []) or full.split(".")[-1] in (tm_list or []))
                if hit and isinstance(child, nn.Linear):
                    if (child.in_features < 64) or (child.out_features < 64):
                        inject(child, prefix=full + "."); continue
                    setattr(module, name, ConvLoRA_Linear(child, cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"],
                                                          k=k, share_conv1d=shared_c1d))
                elif hit and isinstance(child, nn.Conv2d):
                    if getattr(child, "groups", 1) != 1:
                        inject(child, prefix=full + "."); continue
                    setattr(module, name, ConvAdapter_Conv2d(child, cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"],
                                                             k=k, share_dw=shared_dw2,
                                                             safefloor_r=conv_sf_r, safefloor_init=conv_sf_init))
                else:
                    inject(child, prefix=full + ".")
        inject(model)
        _freeze_backbone(model)
        print_trainable_summary(model, "[adapter:conv]")
        return model

    warnings.warn(f"[attach_adapter] Unknown adapter_type {atype!r} — returning model unchanged.")
    return model

# ----------------------------- QLoRA quantization ----------------------------

def _apply_qlora_quantization(model: nn.Module, lowperf: bool = False) -> nn.Module:
    if not HAS_BNB or Linear4bit is None:
        raise RuntimeError("QLoRA requires bitsandbytes with GPU support.")
    if prepare_model_for_kbit_training is None:
        raise RuntimeError("QLoRA requires PEFT's prepare_model_for_kbit_training, which is not available.")
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=("fp4" if lowperf else "nf4"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    setattr(qcfg, "modules_to_not_convert", [
        "prompt_encoder", "mask_decoder", "output_hypernetworks_mlps", "image_encoder.patch_embed",
    ])
    sig = signature(prepare_model_for_kbit_training)
    kwargs = {}
    if "use_gradient_checkpointing" in sig.parameters:
        kwargs["use_gradient_checkpointing"] = False
    model = prepare_model_for_kbit_training(model, **kwargs)

    def _shim_convert_linear_layer(root: nn.Module, cfg: BitsAndBytesConfig) -> nn.Module:
        skip = set(getattr(cfg, "modules_to_not_convert", []) or [])
        compute_dtype = getattr(cfg, "bnb_4bit_compute_dtype", torch.bfloat16)
        quant_type    = getattr(cfg, "bnb_4bit_quant_type", "nf4")
        double_quant  = bool(getattr(cfg, "bnb_4bit_use_double_quant", True))
        replaced = 0
        for name, mod in list(root.named_modules()):
            if isinstance(mod, nn.Linear) and not any(s in name for s in skip):
                parent = root
                parts = name.split(".")
                for p in parts[:-1]: parent = getattr(parent, p)
                attr = parts[-1]
                new = Linear4bit(
                    mod.in_features, mod.out_features,
                    bias=(mod.bias is not None),
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    compress_statistics=double_quant,
                )
                with torch.no_grad():
                    new.weight.copy_(mod.weight.detach())
                    if mod.bias is not None: new.bias.copy_(mod.bias.detach())
                setattr(parent, attr, new)
                replaced += 1
        if replaced == 0:
            warnings.warn("QLoRA shim: found no nn.Linear layers to convert.", RuntimeWarning)
        else:
            print(f"[utils] QLoRA shim converted {replaced} nn.Linear layers -> Linear4bit")
        return root

    used_official = False
    if convert_linear_layer is not None:
        try:
            model = convert_linear_layer(model, qcfg)
            used_official = True
        except TypeError:
            used_official = False
        except Exception as e:
            warnings.warn(f"convert_linear_layer failed ({e}); falling back to shim.")
            used_official = False

    if not used_official:
        model = _shim_convert_linear_layer(model, qcfg)
    if not any(isinstance(l, Linear4bit) for l in model.modules()):
        model = _shim_convert_linear_layer(model, qcfg)

    num_4bit = sum(1 for m in model.modules() if isinstance(m, Linear4bit))
    print(f"[utils] Linear4bit layers: {num_4bit}")
    _check_quant(model)
    return model

# ----------------------------- Network builder -------------------------------

def get_network(
    args,
    net: str,
    *,
    use_gpu: bool = True,
    gpu_device: Union[int, str, torch.device] = 0,
    distribution: str = "none",
    steps: Optional[int] = None,
    num_virtual_tokens: int = 20,
) -> nn.Module:
    device = _resolve_device(gpu_device) if use_gpu else torch.device("cpu")
    net_lc  = (net or "").lower()
    encoder = str(getattr(args, "encoder", "vit_b")).lower()
    atype   = str(getattr(args, "adapter_type", "lora")).lower()

    def _map_encoder_to_embed_dim(enc: str, net_name: str) -> int:
        enc = (enc or "").lower(); net_name = (net_name or "").lower()
        if "vit_h" in enc or enc.endswith("h"): return 1280
        if "vit_l" in enc or enc.endswith("l"): return 1024
        if "vit_b" in enc or enc.endswith("b"): return 768
        if "mobile" in net_name or "efficient" in net_name: return 256
        return 768

    fb = getattr(args, "freq_bands", None)
    if (
        fb is None
        or (isinstance(fb, (int, float)) and int(fb) in (0, -1, 8))
        or (isinstance(fb, str) and fb.strip() in ("", "8"))
    ):
        setattr(args, "freq_bands", _map_encoder_to_embed_dim(encoder, net_lc))

    if net_lc == "sam":
        from models.sam import sam_model_registry
        model = sam_model_registry[args.encoder](args, checkpoint=args.sam_ckpt)
    elif net_lc == "efficient_sam":
        from models.efficient_sam import sam_model_registry
        model = sam_model_registry[args.encoder](args)
    elif net_lc == "mobile_sam":
        from models.MobileSAMv2.mobilesamv2 import sam_model_registry
        model = sam_model_registry[args.encoder](args, checkpoint=args.sam_ckpt)
    else:
        raise ValueError(f"Unsupported network: {net!r}")

    def _infer_embed_dim_from_model(m: nn.Module) -> int:
        try:
            for name, mod in m.named_modules():
                if isinstance(mod, nn.Linear) and name.endswith("attn.qkv"):
                    return int(mod.in_features)
        except Exception:
            pass
        enc = getattr(m, "image_encoder", None)
        if enc is not None and hasattr(enc, "embed_dim"):
            try: return int(getattr(enc, "embed_dim"))
            except Exception: pass
        return _map_encoder_to_embed_dim(encoder, net_lc)

    embed_dim_env = os.environ.get("MEDSAM_EMBED_DIM", "").strip()
    embed_dim = int(embed_dim_env) if embed_dim_env.isdigit() else _infer_embed_dim_from_model(model)

    def _force_retie_space_like(root: nn.Module, new_in: int):
        patched, examined = 0, 0
        for mod in root.modules():
            dfc1 = getattr(mod, "D_fc1", None)
            dfc2 = getattr(mod, "D_fc2", None)
            if isinstance(dfc1, nn.Linear) and isinstance(dfc2, nn.Linear):
                examined += 1
                if int(dfc1.in_features) != int(new_in):
                    mid = int(dfc1.out_features)
                    dev  = dfc1.weight.device
                    dtype= dfc1.weight.dtype
                    new_fc1 = nn.Linear(new_in, mid, bias=(dfc1.bias is not None)).to(device=dev, dtype=dtype)
                    new_fc2 = nn.Linear(mid, new_in, bias=(dfc2.bias is not None)).to(device=dev, dtype=dtype)
                    nn.init.kaiming_uniform_(new_fc1.weight, a=math.sqrt(5))
                    if new_fc1.bias is not None: nn.init.zeros_(new_fc1.bias)
                    nn.init.zeros_(new_fc2.weight)
                    if new_fc2.bias is not None: nn.init.zeros_(new_fc2.bias)
                    setattr(mod, "D_fc1", new_fc1)
                    setattr(mod, "D_fc2", new_fc2)
                    patched += 1
        if patched > 0:
            print(f"[utils] Space‑Adapter retie: fixed {patched} / {examined} modules → in_dim={new_in}")
        else:
            print(f"[utils] Space‑Adapter retie: examined {examined} modules; no mismatches detected")

    _force_retie_space_like(model, embed_dim)

    if atype == "qlora":
        model = _apply_qlora_quantization(model, lowperf=getattr(args, "qlora_lowperf", False))

    model._freq_bands = int(getattr(args, "freq_bands", 0) or 0)

    total_steps = steps or getattr(args, "total_step", None)
    model = attach_adapter(
        model,
        args,
        steps=total_steps,
        num_virtual_tokens=getattr(args, "num_virtual_tokens", num_virtual_tokens),
        target_modules=getattr(args, "target_modules", None),
        share_st=getattr(args, "share_st", True),
        conv_kernel=getattr(args, "conv_kernel", 3),
    )

    if getattr(args, "grad_checkpointing", False) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if use_gpu and torch.cuda.is_available():
        if distribution != "none" and getattr(args, "distributed", None) and str(args.distributed).lower() != "none":
            device_ids = [int(i) for i in str(args.distributed).split(",")]
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
        else:
            model = model.to(device)

    for name, mod in model.named_modules():
        dfc1 = getattr(mod, "D_fc1", None)
        dfc2 = getattr(mod, "D_fc2", None)
        if isinstance(dfc1, nn.Linear) and isinstance(dfc2, nn.Linear):
            if int(dfc1.in_features) != int(embed_dim):
                raise RuntimeError(
                    f"[utils] Space_Adapter still mismatched at {name}: "
                    f"D_fc1.in_features={dfc1.in_features} vs embed_dim={embed_dim}"
                )
    return model

# ----------------------------- Misc helpers ----------------------------------

def _check_quant(model: nn.Module):
    if Linear4bit is None:
        print("[utils] (check) bitsandbytes Linear4bit not available; skip QLoRA check.")
    else:
        if not any(isinstance(l, Linear4bit) for l in model.modules()):
            raise RuntimeError("QLoRA failed — no Linear4bit layers detected")
        print("[utils] ✔ Linear4bit layers detected")

def save_checkpoint(state: dict, is_best: bool, path: str, filename: str = "checkpoint.pth"):
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, filename)
    torch.save(state, fpath)
    if is_best:
        best = os.path.join(path, "model_best.pth")
        try:
            import shutil; shutil.copyfile(fpath, best)
        except Exception:
            pass

# ----------------------------- RoSA runtime helpers --------------------------

def set_rosa_train_topk(model: nn.Module, k_train: int, *, tau: float = None, mix: float = None,
                        rankdrop: float = None, sf_scale: float = None) -> None:
    for m in model.modules():
        if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
            m.set_train_topk(int(k_train))
            if mix is None:
                r = float(max(1, m.r))
                lam = 1.0 - float(k_train) / r
            else:
                lam = float(mix)
            lam = max(getattr(m, "lam_min", 0.25), min(getattr(m, "lam_max", 0.95), lam))
            m.set_train_mix(lam)
            if tau is not None:
                m.hard.set_temperature(float(max(0.1, tau)))
            if rankdrop is not None:
                m.set_rankdrop(float(rankdrop))
            if sf_scale is not None:
                m.set_sf_scale(float(sf_scale))

def l0_regularizer(model: nn.Module) -> torch.Tensor:
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")
    total = torch.tensor(0.0, device=device)
    for m in model.modules():
        if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
            total = total + m.hard.l0().sum()
    return total

def set_rosa_fc2_floor(model: nn.Module, train_frac: float = None, eval_frac: float = None) -> None:
    for m in model.modules():
        if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
            m.set_fc2_floor(train_frac=train_frac, eval_frac=eval_frac)


# ----------------------------- Optimizer (adapter-aware) ---------------------

def _module_is_norm(m: nn.Module) -> bool:
    return isinstance(m, (nn.LayerNorm, nn.GroupNorm, RMSNormLastDim, RMSNorm2d))

def build_adapter_aware_optimizer(
    model: nn.Module,
    lr: float = 1.5e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8,
    fused_ok: bool = True,
):
    decay, no_decay = [], []
    seen = set()

    def _add(param, name, module):
        if not getattr(param, "requires_grad", False):
            return
        if (
            name.endswith("bias")
            or _module_is_norm(module)
            or any(tag in name for tag in [".rank_gate", ".log_alpha", ".spec_gain", ".adapter_gain", ".sf_gate", ".delta_s"])
            or any(tag in name for tag in (".sfA.", ".sfB."))
        ):
            bucket = no_decay
        else:
            bucket = decay
        pid = id(param)
        if pid in seen:
            return
        seen.add(pid)
        bucket.append(param)

    for mod_name, mod in model.named_modules():
        for p_name, p in mod.named_parameters(recurse=False):
            _add(p, f"{mod_name}.{p_name}" if mod_name else p_name, mod)

    for p in model.parameters():
        if id(p) in seen or not getattr(p, "requires_grad", False):
            continue
        (no_decay if p.ndim == 1 else decay).append(p)
        seen.add(id(p))

    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    use_fused = False
    if fused_ok and hasattr(torch.optim, "AdamW"):
        try:
            use_fused = bool(torch.cuda.is_available())
        except Exception:
            use_fused = False

    optimizer = optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, fused=use_fused) \
        if hasattr(optim, "AdamW") else optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)

    decay_num  = sum(p.numel() for p in decay)
    nodec_num  = sum(p.numel() for p in no_decay)
    total_num  = decay_num + nodec_num
    print(f"[optim] AdamW lr={lr} wd={weight_decay} fused={use_fused} | "
          f"trainable={total_num/1e6:.2f}M (decay={decay_num/1e6:.2f}M, no_decay={nodec_num/1e6:.2f}M)")
    return optimizer

# ----------------------------- Trainer-side helpers --------------------------

def schedule_rosa_from_args(model, step: int, total_steps: int, args):
    k0   = int(getattr(args, "rosa_k_train_init", 0))
    kf   = int(getattr(args, "rosa_k_train_final", 96))
    wfr  = float(getattr(args, "rosa_k_warmup_frac", 0.25))
    # accept tau or temp aliases
    tau0 = float(getattr(args, "rosa_tau_init",  getattr(args, "rosa_temp_init", 2.0/3.0)))
    tauf = float(getattr(args, "rosa_tau_final", getattr(args, "rosa_temp_final", 2.0/3.0)))

    prog = step / max(1, total_steps)

    # (1) Top‑K warm‑up to kf by warm‑up fraction
    k_now = int(k0 + (kf - k0) * min(1.0, prog / max(1e-8, wfr))) if prog < wfr else kf

    # (2) Gate temperature schedule
    tau_now = tau0 + (tauf - tau0) * min(1.0, prog)

    # (3) Rank‑drop schedule
    p0  = float(getattr(args, "rosa_rankdrop_init", 0.10))
    pf  = float(getattr(args, "rosa_rankdrop_final", 0.00))
    p_now = p0 + (pf - p0) * min(1.0, prog)

    # (4) Safe‑floor fade (unchanged logic)
    sf_scale = 1.0 if prog < 0.5 else (1.0 - 0.5 * ((prog - 0.5) / 0.5))

    # (5) Optional explicit teacher‑mix schedule (lam); if unset, module default applies
    lam0 = getattr(args, "rosa_mix_init",  None)
    lamf = getattr(args, "rosa_mix_final", None)
    mix_now = None
    if (lam0 is not None) or (lamf is not None):
        lam0 = 0.90 if lam0 is None else float(lam0)
        lamf = 0.20 if lamf is None else float(lamf)
        mix_now = lam0 + (lamf - lam0) * min(1.0, prog)

    # (6) RoSA‑OPF: fc2 floor ramp (train) + fixed eval floor
    # NOTE: we use defaults here so you DON'T need cfg.py changes; CLI overrides optional.
    fc2_floor_train_init = float(getattr(args, "rosa_fc2_floor_train_init", 0.12))
    fc2_floor_train_final = float(getattr(args, "rosa_fc2_floor_train_final", 0.25))
    if prog < wfr:
        floor_now = fc2_floor_train_init + (fc2_floor_train_final - fc2_floor_train_init) * min(1.0, prog / max(1e-8, wfr))
    else:
        floor_now = fc2_floor_train_final
    fc2_floor_eval = float(getattr(args, "rosa_fc2_floor_eval", 0.33))
    set_rosa_fc2_floor(model, train_frac=floor_now, eval_frac=fc2_floor_eval)

    # (7) Apply
    set_rosa_train_topk(model, k_train=k_now, tau=tau_now, mix=mix_now,
                        rankdrop=p_now, sf_scale=sf_scale)

# ---- AdaLoRA shim -----------------------------------------------------------

def adalora_update_and_allocate_if_present(model: nn.Module,
                                           step: Optional[int] = None,
                                           total_step: Optional[int] = None,
                                           **kwargs) -> bool:
    if model is None:
        return False
    try:
        handled = False
        for m in model.modules():
            cls_name = m.__class__.__name__.lower()
            if "adalora" in cls_name and hasattr(m, "update_and_allocate"):
                try:
                    upd = getattr(m, "update_and_allocate")
                    sig = signature(upd)
                    call_kwargs = {}
                    if "step" in sig.parameters:
                        call_kwargs["step"] = step
                    if "total_step" in sig.parameters:
                        call_kwargs["total_step"] = total_step
                    for k, v in kwargs.items():
                        if k in sig.parameters:
                            call_kwargs[k] = v
                    upd(**call_kwargs)
                    handled = True
                except Exception as e:
                    warnings.warn(f"AdaLoRA update_and_allocate failed: {e}")
        return handled
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────
# 6.  Data loading helpers
# ──────────────────────────────────────────────────────────────
def get_decath_loader(args):
    """Return MONAI dataloaders for Decathlon dataset."""
    # Note: `DEVICE` is global.
    train_tf = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=DEVICE, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(args.roi_size, args.roi_size, args.chunk),
            pos=1, neg=1, num_samples=args.num_sample,
            image_key="image", image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])
    val_tf = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=DEVICE, track_meta=True),
    ])

    data_dir   = args.data_path
    split_json = os.path.join(data_dir, "dataset_0.json")

    train_files = load_decathlon_datalist(split_json, True, "training")
    val_files   = load_decathlon_datalist(split_json, True, "validation")

    train_ds = CacheDataset(train_files, transform=train_tf, cache_num=24, cache_rate=1.0, num_workers=8)
    val_ds   = CacheDataset(val_files,   transform=val_tf,   cache_num=2,  cache_rate=1.0, num_workers=0)

    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=args.b, shuffle=True)
    val_loader   = ThreadDataLoader(val_ds,   num_workers=0, batch_size=1)

    set_track_meta(False)
    return train_loader, val_loader, train_tf, val_tf, train_files, val_files

# ──────────────────────────────────────────────────────────────
# 7.  Losses & schedulers
# ──────────────────────────────────────────────────────────────
def cka_loss(gram_A: torch.Tensor, gram_B: torch.Tensor) -> torch.Tensor:
    """Centered kernel alignment loss between two Gram matrices."""
    scaled_hsic = torch.dot(gram_A.flatten(), gram_B.flatten())
    return scaled_hsic / (gram_A.norm() * gram_B.norm())

class WarmUpLR(_LRScheduler):
    """Linear warm‑up scheduler (per batch)."""
    def __init__(self, optimizer, total_iters: int, last_epoch: int = -1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

# ──────────────────────────────────────────────────────────────
# 8.  Image I/O utilities (make_grid / save_image)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
) -> torch.Tensor:
    """Re‑implementation of `torchvision.utils.make_grid` (CUDA‑safe)."""
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # Convert list to 4‑D minibatch
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single H×W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # single‑channel → 3‑channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 4 and tensor.size(1) == 1:  # B×1×H×W
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize:
        tensor = tensor.clone()  # avoid in‑place
        def _norm_img(img, lo, hi):
            img.clamp_(lo, hi).add_(-lo).div_(max(hi - lo, 1e-5))
        def _norm_range(t, vr):
            _norm_img(t, vr[0], vr[1]) if vr else _norm_img(t, float(t.min()), float(t.max()))
        if scale_each:
            for t in tensor: _norm_range(t, value_range)
        else:
            _norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(nmaps / xmaps))
    h, w  = tensor.size(2) + padding, tensor.size(3) + padding
    grid  = tensor.new_full((tensor.size(1), h * ymaps + padding, w * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps: break
            grid.narrow(1, y*h + padding, h - padding)\
                .narrow(2, x*w + padding, w - padding)\
                .copy_(tensor[k])
            k += 1
    return grid

@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """Save Tensor or list[T] to disk (uses `make_grid`)."""
    grid  = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255)\
                 .permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    Image.fromarray(ndarr).save(fp, format=format)

# ──────────────────────────────────────────────────────────────
# 9.  Logging & checkpoint helpers
# ──────────────────────────────────────────────────────────────
def create_logger(log_dir: str, phase: str = "train"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{time.strftime('%Y-%m-%d-%H-%M')}_{phase}.log"
    final_log_file = os.path.join(log_dir, log_file)
    logging.basicConfig(filename=final_log_file, format="%(asctime)-15s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger("").addHandler(logging.StreamHandler())
    return logger

def set_log_dir(root_dir: str, exp_name: str):
    os.makedirs(root_dir, exist_ok=True)
    exp_path = os.path.join(root_dir, exp_name)
    timestamp = datetime.now(dateutil.tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")
    prefix = f"{exp_path}_{timestamp}"
    os.makedirs(prefix)

    paths = {
        "prefix"     : prefix,
        "ckpt_path"  : os.path.join(prefix, "Model"),
        "log_path"   : os.path.join(prefix, "Log"),
        "sample_path": os.path.join(prefix, "Samples"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def save_checkpoint(states: dict, is_best: bool, output_dir: str, filename: str = "checkpoint.pth"):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, "checkpoint_best.pth"))

# ──────────────────────────────────────────────────────────────
# 10.  Statistics & metrics
# ──────────────────────────────────────────────────────────────
class RunningStats:
    """Welford’s online algorithm over a sliding window."""
    def __init__(self, win_size: int):
        self.WIN_SIZE   = win_size
        self.window     = collections.deque(maxlen=win_size)
        self.mean       = 0.0
        self.run_var    = 0.0

    def clear(self): self.__init__(self.WIN_SIZE)
    def is_full(self): return len(self.window) == self.WIN_SIZE

    def push(self, x: float):
        if self.is_full():
            x_rem      = self.window.popleft()
            self.window.append(x)
            old_mean   = self.mean
            self.mean += (x - x_rem) / self.WIN_SIZE
            self.run_var += (x + x_rem - old_mean - self.mean) * (x - x_rem)
        else:
            self.window.append(x)
            delta = x - self.mean
            self.mean += delta / len(self.window)
            self.run_var += delta * (x - self.mean)

    def get_mean(self): return self.mean if self.window else 0.0
    def get_var(self):  return self.run_var / len(self.window) if len(self.window) > 1 else 0.0
    def get_std(self):  return math.sqrt(self.get_var())
    def get_all(self):  return list(self.window)
    def __str__(self):  return f"Current window values: {list(self.window)}"

def iou(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Intersection‑over‑Union for binary masks (batch mean)."""
    if outputs.dim() == 4: outputs = outputs.squeeze(1)
    if labels.dim()  == 4: labels  = labels.squeeze(1)
    smooth = 1e-6
    intersection = (outputs * labels).sum(dim=(1, 2))
    union        = (outputs + labels - outputs * labels).sum(dim=(1, 2))
    return ((intersection + smooth) / (union + smooth)).mean()

class DiceCoeff(Function):
    """Dice coefficient for individual examples (autograd‑friendly)."""
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 1e-4
        self.inter = torch.dot(input.flatten(), target.flatten())
        self.union = input.sum() + target.sum() + eps
        return (2 * self.inter + eps) / self.union

    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) / (self.union ** 2)
        return grad_input, grad_target

def dice_coeff(input: torch.Tensor, target: torch.Tensor):
    """Dice coefficient for batches (mean of examples)."""
    s = torch.zeros(1, device=input.device if input.is_cuda else "cpu")
    for i, (inp, tgt) in enumerate(zip(input, target)):
        s += DiceCoeff().apply(inp, tgt)
    return s / (i + 1)

def hd95(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Robust 95‑percentile Hausdorff distance (2‑D or 3‑D).

    • Both masks empty → 0.0
    • One mask empty  → image diagonal
    """
    if preds.dim() == 3:
        preds  = preds.unsqueeze(1)
        labels = labels.unsqueeze(1)

    preds_bool  = preds  > 0.5
    labels_bool = labels > 0.5
    hd_batch = []
    for p, l in zip(preds_bool, labels_bool):
        dev = p.device
        if not p.any() and not l.any():
            hd_batch.append(torch.tensor(0.0, device=dev)); continue
        if not p.any() or not l.any():
            diag = torch.tensor(p.shape[1:], dtype=torch.float32, device=dev)
            hd_batch.append(torch.linalg.norm(diag));        continue
        d = compute_hausdorff_distance(p.unsqueeze(0), l.unsqueeze(0), percentile=95).squeeze()
        hd_batch.append(torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).to(dev))
    return torch.stack(hd_batch).mean().item()
# =========================================================================

'''parameter'''
def para_image(w, h=None, img = None, mode = 'multi', seg = None, sd=None, batch=None,
          fft = False, channels=None, init = None):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    if init is not None:
        param_f = init_image
        params, maps_f = param_f(init)
    else:
        params, maps_f = param_f(shape, sd=sd)
    if mode == 'multi':
        output = to_valid_out(maps_f,img,seg)
    elif mode == 'seg':
        output = gene_out(maps_f,img)
    elif mode == 'raw':
        output = raw_out(maps_f,img)
    return params, output

def to_valid_out(maps_f,img,seg): #multi-rater
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        maps = torch.nn.Softmax(dim = 1)(maps)
        final_seg = torch.multiply(seg,maps).sum(dim = 1, keepdim = True)
        return torch.cat((img,final_seg),1)
        # return torch.cat((img,maps),1)
    return inner

def gene_out(maps_f,img): #pure seg
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return torch.cat((img,maps),1)
        # return torch.cat((img,maps),1)
    return inner

def raw_out(maps_f,img): #raw
    def inner():
        maps = maps_f()
        maps = maps.to(device = img.device)
        # maps = torch.nn.Sigmoid()(maps)
        return maps
        # return torch.cat((img,maps),1)
    return inner    


class CompositeActivation(torch.nn.Module):

    def forward(self, x):
        x = torch.atan(x)
        return torch.cat([x/0.67, (x*x)/0.6], 1)
        # return x


def cppn(args, size, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):

    r = 3 ** 0.5

    coord_range = torch.linspace(-r, r, size)
    x = coord_range.view(-1, 1).repeat(1, coord_range.size(0))
    y = coord_range.view(1, -1).repeat(coord_range.size(0), 1)

    input_tensor = torch.stack([x, y], dim=0).unsqueeze(0).repeat(batch,1,1,1).to(device)

    layers = []
    kernel_size = 1
    for i in range(num_layers):
        out_c = num_hidden_channels
        in_c = out_c * 2 # * 2 for composite activation
        if i == 0:
            in_c = 2
        if i == num_layers - 1:
            out_c = num_output_channels
        layers.append(('conv{}'.format(i), torch.nn.Conv2d(in_c, out_c, kernel_size)))
        if normalize:
            layers.append(('norm{}'.format(i), torch.nn.InstanceNorm2d(out_c)))
        if i < num_layers - 1:
            layers.append(('actv{}'.format(i), activation_fn()))
        else:
            layers.append(('output', torch.nn.Sigmoid()))

    # Initialize model
    net = torch.nn.Sequential(OrderedDict(layers)).to(device)
    # Initialize weights
    def weights_init(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0, np.sqrt(1/module.in_channels))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    net.apply(weights_init)
    # Set last conv2d layer's weights to 0
    torch.nn.init.zeros_(dict(net.named_children())['conv{}'.format(num_layers - 1)].weight)
    outimg = raw_out(lambda: net(input_tensor),img) if args.netype == 'raw' else to_valid_out(lambda: net(input_tensor),img,seg)
    return net.parameters(), outimg

def get_siren(args):
    wrapper = get_network(args, 'siren', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    '''load init weights'''
    checkpoint = torch.load('./logs/siren_train_init_2022_08_19_21_00_16/Model/checkpoint_best.pth')
    wrapper.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    '''load prompt'''
    checkpoint = torch.load('./logs/vae_standard_refuge1_2022_08_21_17_56_49/Model/checkpoint500')
    vae = get_network(args, 'vae', use_gpu=args.gpu, gpu_device=torch.device('cuda', args.gpu_device), distribution = args.distributed)
    vae.load_state_dict(checkpoint['state_dict'],strict=False)
    '''end'''

    return wrapper, vae


def siren(args, wrapper, vae, img = None, seg = None, batch=None, num_output_channels=1, num_hidden_channels=128, num_layers=8,
         activation_fn=CompositeActivation, normalize=False, device = "cuda:0"):
    vae_img = torchvision.transforms.Resize(64)(img)
    latent = vae.encoder(vae_img).view(-1).detach()
    outimg = raw_out(lambda: wrapper(latent = latent),img) if args.netype == 'raw' else to_valid_out(lambda: wrapper(latent = latent),img,seg)
    # img = torch.randn(1, 3, 256, 256)
    # loss = wrapper(img)
    # loss.backward()

    # # after much training ...
    # # simply invoke the wrapper without passing in anything

    # pred_img = wrapper() # (1, 3, 256, 256)
    return wrapper.parameters(), outimg
        

'''adversary'''
def render_vis(
    args,
    model,
    objective_f,
    real_img,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(256,),
    verbose=True,
    preprocess=True,
    progress=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    fixed_image_size=None,
    label = 1,
    raw_img = None,
    prompt = None
):
    if label == 1:
        sign = 1
    elif label == 0:
        sign = -1
    else:
        print('label is wrong, label is',label)
    if args.reverse:
        sign = -sign
    if args.multilayer:
        sign = 1

    '''prepare'''
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y, %H:%M:%S")

    netD, optD = pre_d()
    '''end'''

    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()
    
    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-1)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = []
    transforms = transforms.copy()

    # Upsample images smaller than 224
    image_shape = image_f().shape

    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss of ad: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            optimizer.zero_grad()
            try:
                model(transform_f(image_f()))
            except RuntimeError as ex:
                if i == 1:
                    # Only display the warning message
                    # on the first iteration, no need to do that
                    # every iteration
                    warnings.warn(
                        "Some layers could not be computed because the size of the "
                        "image is not big enough. It is fine, as long as the non"
                        "computed layers are not used in the objective function"
                        f"(exception details: '{ex}')"
                    )
            if args.disc:
                '''dom loss part'''
                # content_img = raw_img
                # style_img = raw_img
                # precpt_loss = run_precpt(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, transform_f(image_f()))
                for p in netD.parameters():
                    p.requires_grad = True
                for _ in range(args.drec):
                    netD.zero_grad()
                    real = real_img
                    fake = image_f()
                    # for _ in range(6):
                    #     errD, D_x, D_G_z1 = update_d(args, netD, optD, real, fake)

                    # label = torch.full((args.b,), 1., dtype=torch.float, device=device)
                    # label.fill_(1.)
                    # output = netD(fake).view(-1)
                    # errG = nn.BCELoss()(output, label)
                    # D_G_z2 = output.mean().item()
                    # dom_loss = err
                    one = torch.tensor(1, dtype=torch.float)
                    mone = one * -1
                    one = one.cuda(args.gpu_device)
                    mone = mone.cuda(args.gpu_device)

                    d_loss_real = netD(real)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)

                    d_loss_fake = netD(fake)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)

                    # Train with gradient penalty
                    gradient_penalty = calculate_gradient_penalty(netD, real.data, fake.data)
                    gradient_penalty.backward()


                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real - d_loss_fake
                    optD.step()

                # Generator update
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation

                fake_images = image_f()
                g_loss = netD(fake_images)
                g_loss = -g_loss.mean()
                dom_loss = g_loss
                g_cost = -g_loss

                if i% 5 == 0:
                    print(f' loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
                    print(f'Generator g_loss: {g_loss}')
                '''end'''



            '''ssim loss'''

            '''end'''

            if args.disc:
                loss = sign * objective_f(hook) + args.pw * dom_loss
                # loss = args.pw * dom_loss
            else:
                loss = sign * objective_f(hook)
                # loss = args.pw * dom_loss

            loss.backward()

            # #video the images
            # if i % 5 == 0:
            #     print('1')
            #     image_name = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
            #     img_path = os.path.join(args.path_helper['sample_path'], str(image_name))
            #     export(image_f(), img_path)
            # #end
            # if i % 50 == 0:
            #     print('Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #       % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            optimizer.step()
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                # if verbose:
                #     print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                if save_image:
                    na = image_name[0].split('\\')[-1].split('.')[0] + '_' + str(i) + '.png'
                    na = date_time + na
                    outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
                    img_path = os.path.join(outpath, str(na))
                    export(image_f(), img_path)
                
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))

    if save_image:
        na = image_name[0].split('\\')[-1].split('.')[0] + '.png'
        na = date_time + na
        outpath = args.quickcheck if args.quickcheck else args.path_helper['sample_path']
        img_path = os.path.join(outpath, str(na))
        export(image_f(), img_path)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
    return image_f()


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def export(tensor, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tensor.size(1)
    # if c == 7:
    #     for i in range(c):
    #         w_map = tensor[:,i,:,:].unsqueeze(1)
    #         w_map = tensor_to_img_array(w_map).squeeze()
    #         w_map = (w_map * 255).astype(np.uint8)
    #         image_name = image_name[0].split('/')[-1].split('.')[0] + str(i)+ '.png'
    #         wheat = sns.heatmap(w_map,cmap='coolwarm')
    #         figure = wheat.get_figure()    
    #         figure.savefig ('./fft_maps/weightheatmap/'+str(image_name), dpi=400)
    #         figure = 0
    # else:
    if c == 3:
        vutils.save_image(tensor, fp = img_path)
    else:
        image = tensor[:,0:3,:,:]
        w_map = tensor[:,-1,:,:].unsqueeze(1)
        image = tensor_to_img_array(image)
        w_map = 1 - tensor_to_img_array(w_map).squeeze()
        # w_map[w_map==1] = 0
        assert len(image.shape) in [
            3,
            4,
        ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
        # Change dtype for PIL.Image
        image = (image * 255).astype(np.uint8)
        w_map = (w_map * 255).astype(np.uint8)

        Image.fromarray(w_map,'L').save(img_path)


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None


    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output


    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    features = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with lucent.modelzoo.util.get_model_layers(model)."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: model.to(device).eval(). See README for example."
        return out

    return hook

def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None, boxes = None):
    
    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2: # for REFUGE multi mask output
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    elif c > 2: # for multi-class segmentation > 2 classes
        preds = []
        gts = []
        for i in range(0, c):
            pred = pred_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            preds.append(pred)
            gt = gt_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            gts.append(gt)
        tup = [imgs[:row_num,:,:,:]] + preds + gts
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if points != None:
            for i in range(b):
                if args.thd:
                    ps = np.round(points.cpu()/args.roi_size * args.out_size).to(dtype = torch.int)
                else:
                    ps = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                # gt_masks[i,:,points[i,0]-5:points[i,0]+5,points[i,1]-5:points[i,1]+5] = torch.Tensor([255, 0, 0]).to(dtype = torch.float32, device = torch.device('cuda:' + str(dev)))
                for p in ps:
                    gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                    gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                    gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4
        if boxes is not None:
            for i in range(b):
                # the next line causes: ValueError: Tensor uint8 expected, got torch.float32
                # imgs[i, :] = torchvision.utils.draw_bounding_boxes(imgs[i, :], boxes[i])
                # until TorchVision 0.19 is released (paired with Pytorch 2.4), apply this workaround:
                img255 = (imgs[i] * 255).byte()
                img255 = torchvision.utils.draw_bounding_boxes(img255, boxes[i].reshape(-1, 4), colors="red")
                img01 = img255 / 255
                # torchvision.utils.save_image(img01, save_path + "_boxes.png")
                imgs[i, :] = img01
        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    elif c > 2: # for multi-class segmentation > 2 classes
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred = vpred_cpu[:,i,:,:].numpy().astype('int32')
                mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
        
                '''iou for numpy'''
                ious[i] += iou(pred,mask)

                '''dice for torch'''
                dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
            
        return tuple(np.array(ious + dices) / len(threshold)) # tuple has a total number of c * 2
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)

# @objectives.wrap_objective()
def dot_compare(layer, batch=1, cossim_pow=0):
  def inner(T):
    dot = (T(layer)[batch] * T(layer)[0]).sum()
    mag = torch.sqrt(torch.sum(T(layer)[0]**2))
    cossim = dot/(1e-6 + mag)
    return -dot * cossim ** cossim_pow
  return inner

def init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def pre_d():
    netD = Discriminator(3).to(device)
    # netD.apply(init_D)
    beta1 = 0.5
    dis_lr = 0.00002
    optimizerD = optim.Adam(netD.parameters(), lr=dis_lr, betas=(beta1, 0.999))
    return netD, optimizerD

def update_d(args, netD, optimizerD, real, fake):
    criterion = nn.BCELoss()

    label = torch.full((args.b,), 1., dtype=torch.float, device=device)
    output = netD(real).view(-1)
    # Calculate loss on all-real batch
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    label.fill_(0.)
    # Classify all fake batch with D
    output = netD(fake.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    optimizerD.step()

    return errD, D_x, D_G_z1

def calculate_gradient_penalty(netD, real_images, fake_images):
    eta = torch.FloatTensor(args.b,1,1,1).uniform_(0,1)
    eta = eta.expand(args.b, real_images.size(1), real_images.size(2), real_images.size(3)).to(device = device)

    interpolated = (eta * real_images + ((1 - eta) * fake_images)).to(device = device)

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(
                                prob_interpolated.size()).to(device = device),
                            create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


def random_click(mask, point_labels = 1):
    # check if all masks are black
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        point_labels = max_label
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    return point_labels, indices[np.random.randint(len(indices))]


def generate_click_prompt(img, msk, pt_label=1):
    """
    If 'msk' is 5D (e.g. [B, C, H, W, D]), we treat it as 3D volumes.
    If 'msk' is 4D (e.g. [B, C, H, W]), we treat it as 2D slices (D=1).
    Returns:
      img (unchanged),
      pt (point coordinates) shape: [B, 2, D],
      msk (new mask) shape: [B, 1, H, W, D]
    """
    pt_list = []
    msk_list = []

    dims = msk.ndim
    if dims == 5:
        b, c, h, w, d = msk.size()
        # remove channel dimension
        msk = msk[:, 0, :, :, :]  # shape [B, H, W, D]
    elif dims == 4:
        b, c, h, w = msk.size()
        d = 1
        msk = msk[:, 0, :, :]  # shape [B, H, W]
        msk = msk.unsqueeze(-1)  # shape [B, H, W, 1]
    else:
        raise ValueError(f"Unsupported input shape {msk.size()}")

    # Now msk is [B, H, W, D]
    for i in range(d):
        pt_list_s = []
        msk_list_s = []
        for j in range(b):
            msk_s = msk[j, :, :, i]  # shape [H, W]
            indices = torch.nonzero(msk_s)
            if indices.size(0) == 0:
                random_index = torch.randint(0, h, (2,)).to(msk.device)
                new_s = msk_s
            else:
                random_index = random.choice(indices)
                label = msk_s[random_index[0], random_index[1]]
                new_s = torch.zeros_like(msk_s)
                new_s = (msk_s == label).float()
            pt_list_s.append(random_index)
            msk_list_s.append(new_s)

        pts = torch.stack(pt_list_s, dim=0)    # shape [B, 2]
        msks = torch.stack(msk_list_s, dim=0)  # shape [B, H, W]
        pt_list.append(pts)
        msk_list.append(msks)

    pt = torch.stack(pt_list, dim=-1)   # shape [B, 2, D]
    msk = torch.stack(msk_list, dim=-1) # shape [B, H, W, D]
    msk = msk.unsqueeze(1)              # shape [B, 1, H, W, D]
    return img, pt, msk


def random_box(multi_rater):
    max_value = torch.max(multi_rater[:, 0, :, :], dim=0)[0]
    max_value_position = torch.nonzero(max_value)
    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]
    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))

    x_min = random.choice(np.arange(x_min - 10, x_min + 11))
    x_max = random.choice(np.arange(x_max - 10, x_max + 11))
    y_min = random.choice(np.arange(y_min - 10, y_min + 11))
    y_max = random.choice(np.arange(y_max - 10, y_max + 11))

    return x_min, x_max, y_min, y_max
