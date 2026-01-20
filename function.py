# Ramtin Mojtahedi
# function.py
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
from tqdm import tqdm

import cfg
import models.sam.utils.transforms as samtrans
import pytorch_ssim
from conf import settings
from utils import *
from utils import iou, dice_coeff, hd95
from utils import adalora_update_and_allocate_if_present


# ─── Global definitions ──────────────────────────────────────────────────────
args = cfg.parse_args()
GPUdevice = torch.device("cuda", args.gpu_device)

# Weighted BCE (or any other). Adjust if needed.
pos_weight = torch.ones([1], device=GPUdevice) * 2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# DiceCELoss (MONAI) for 1-channel or multi-channel seg
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

# AMP GradScaler (new API in torch 2.x)
scaler = torch.amp.GradScaler("cuda")

# For speed
torch.backends.cudnn.benchmark = True


# ------------------ Training Function ------------------ #
def train_sam(
    args,
    net: nn.Module,
    optimizer,
    train_loader,
    epoch: int,
    writer,
    schedulers=None,
    vis: int = 50,
):
    """
    One epoch of SAM(-adapter) training with random/provided prompts.

    - Uses BCE/Dice criterion (criterion_G) or MONAI DiceCELoss for 3D.
    - Optional L1 sparsity on modules exposing `.gate` via --lambda_gate (default 0.0).
    - Calls AdaLoRA per-step rank allocator when adapter_type == "adalora".
    """
    net.train()
    GPUdevice = torch.device("cuda", args.gpu_device)

    # AMP scaler: use global if present, else make a local one
    scaler_local = globals().get("scaler", None)
    if scaler_local is None:
        scaler_local = torch.cuda.amp.GradScaler()

    optimizer.zero_grad()

    # loss function
    if getattr(args, "thd", False):
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    else:
        lossfunc = criterion_G

    epoch_loss = 0.0
    step_idx = 0
    steps_per_epoch = len(train_loader)

    # read sparsity weight (0.0 = disabled / fair comparison)
    lambda_gate = float(getattr(args, "lambda_gate", 0.0) or 0.0)

    with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", unit="img") as pbar:
        for pack in train_loader:
            imgs = pack["image"].to(dtype=torch.float32, device=GPUdevice)
            masks = pack["label"].to(dtype=torch.float32, device=GPUdevice)

            # prompts
            if ("pt" not in pack) or ("p_label" not in pack):
                imgs, (pt_coords, pt_labels), masks = generate_single_point_prompt(imgs, masks)
            else:
                pt_coords = pack["pt"].to(device=GPUdevice)
                pt_labels = pack["p_label"].to(device=GPUdevice)

            with torch.cuda.amp.autocast(enabled=False):
                # Encode image
                image_embeddings = net.image_encoder(imgs)

                # Prompt encoder (tuple or single tensor depending on backend)
                if args.net in ["sam", "mobile_sam", "efficient_sam"]:
                    out = net.prompt_encoder(points=(pt_coords, pt_labels), boxes=None, masks=None)
                    if isinstance(out, (tuple, list)):
                        se, de = out if len(out) == 2 else (out, None)
                    else:
                        se, de = out, None
                else:
                    se, de = None, None

                # Decode
                if args.net in {"sam", "mobile_sam"}:
                    pred, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )
                elif args.net == "efficient_sam":
                    pred, _ = net.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        multimask_output=False,
                    )
                else:
                    pred = image_embeddings  # fallback

                # Resize & loss
                pred = F.interpolate(pred, size=(args.out_size, args.out_size))
                loss = lossfunc(pred, masks)

                # Optional sparsity term on adapters exposing `.gate`
                if lambda_gate > 0.0:
                    sparsity = 0.0
                    for m in net.modules():
                        if hasattr(m, "gate"):
                            sparsity = sparsity + torch.abs(m.gate).sum()
                    loss = loss + lambda_gate * sparsity

            # ---- backward / step -------------------------------------------------
            scaler_local.scale(loss).backward()

            # unscale + clip
            scaler_local.unscale_(optimizer)
            clip_val = getattr(args, "max_grad_norm", None)
            if clip_val and clip_val > 0:
                params = (net.module.parameters() if hasattr(net, "module") else net.parameters())
                torch.nn.utils.clip_grad_norm_(params, clip_val)

            # optimizer step
            scaler_local.step(optimizer)
            scaler_local.update()

            # AdaLoRA: per-step rank allocation (after step, before zero_grad)
            if str(getattr(args, "adapter_type", "")).lower() == "adalora":
                model_for_alloc = net.module if hasattr(net, "module") else net
                global_step = epoch * steps_per_epoch + step_idx  # 0-based
                adalora_update_and_allocate_if_present(model_for_alloc, global_step)

            optimizer.zero_grad()
            # ---------------------------------------------------------------------

            epoch_loss += float(loss.item())
            step_idx += 1
            pbar.set_postfix(**{"loss (batch)": float(loss.item())})
            pbar.update()

            # optional visualization
            if vis and isinstance(vis, int) and (step_idx % vis == 0):
                outpath = os.path.join(
                    args.path_helper["sample_path"], f"train_epoch{epoch}_step{step_idx}.jpg"
                )
                vis_image(imgs, pred, masks, outpath, reverse=False, points=pt_coords)

    return epoch_loss / max(1, step_idx)


# ------------------ Validation Function ------------------ #
def validation_sam(
    args,
    val_data,  # dict (single batch) OR DataLoader
    epoch: int,
    net: torch.nn.Module,
    writer,
    prompt_type: Optional[str] = None,
    clean_dir: bool = True,
    visualize_prediction: bool = False,
):
    """
    Evaluate segmentation under different prompt regimes.
    Returns: (loss, (iou, dice, hd95))
    """
    net.eval()
    GPUdevice = torch.device("cuda", args.gpu_device)

    def _forward(imgs, masks, points_to_use, boxes_to_use):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            img_emb = net.image_encoder(imgs)

            # Prompt encoder
            if args.net in {"sam", "mobile_sam"}:
                se, de = net.prompt_encoder(points=points_to_use, boxes=boxes_to_use, masks=None)
            elif args.net == "efficient_sam":
                out = net.prompt_encoder(points=points_to_use, boxes=boxes_to_use, masks=None)
                se, de = out if isinstance(out, (tuple, list)) else (out, None)
            else:
                se = de = None

            # Mask decoder
            if args.net in {"sam", "mobile_sam", "efficient_sam"}:
                pred, _ = net.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                )
            else:
                pred = img_emb

            pred = F.interpolate(pred, size=(args.out_size, args.out_size))

            # loss (for logging)
            pos_weight = torch.ones([1], device=GPUdevice) * 2
            lossfunc = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_val = lossfunc(pred, masks)

            # metrics
            thresholds = (0.1, 0.3, 0.5, 0.7, 0.9)
            iou_val, dice_val, hd95_val = eval_seg(pred, masks, thresholds)

        return loss_val.item(), (iou_val, dice_val, hd95_val), pred

    # single-batch mode
    if isinstance(val_data, dict):
        pack = val_data
        imgs = pack["image"].to(dtype=torch.float32, device=GPUdevice)
        masks = pack["label"].to(dtype=torch.float32, device=GPUdevice)

        points_to_use = boxes_to_use = None
        if prompt_type == "single_point":
            imgs, (pt, pl), masks = generate_single_point_prompt(imgs, masks)
            points_to_use = (pt, pl)
        elif prompt_type == "multi_point":
            imgs, (pt, pl), masks = generate_multi_point_prompt(imgs, masks, num_points=3)
            points_to_use = (pt, pl)
        elif prompt_type == "bbox_05":
            imgs, boxes_to_use, masks = generate_bbox_prompt(imgs, masks, ratio=0.50)
        elif prompt_type == "bbox_075":
            imgs, boxes_to_use, masks = generate_bbox_prompt(imgs, masks, ratio=0.75)
        elif prompt_type in {None, "none", "noprompt", "no_prompt"}:
            pass
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        loss_val, metric_tup, pred = _forward(imgs, masks, points_to_use, boxes_to_use)

        if visualize_prediction:
            outpath = os.path.join(
                args.path_helper["sample_path"], f"val_epoch{epoch}_{prompt_type or 'none'}.jpg"
            )
            vis_image(imgs, (pred > 0.5).float(), masks, outpath, reverse=False)

        return loss_val, metric_tup

    # DataLoader mode (dataset average)
    total_loss, acc_metrics, n_batches = 0.0, None, 0
    with tqdm(
        total=len(val_data),
        desc=f"Valid (prompt={prompt_type})",
        unit="batch",
        leave=False,
    ) as pbar:
        for pack in val_data:
            batch_loss, batch_metrics = validation_sam(
                args,
                pack,
                epoch,
                net,
                writer,
                prompt_type=prompt_type,
                visualize_prediction=False,
            )
            total_loss += batch_loss
            if acc_metrics is None:
                acc_metrics = list(batch_metrics)
            else:
                for i, v in enumerate(batch_metrics):
                    acc_metrics[i] += v
            n_batches += 1
            pbar.update()

    avg_loss = total_loss / max(1, n_batches)
    avg_metrics = tuple(m / max(1, n_batches) for m in acc_metrics)
    return avg_loss, avg_metrics


# ------------- Prompt Generation Functions ------------- #
def generate_single_point_prompt(imgs, masks):
    """
    Return (imgs, (pt_coords[B,1,2], pt_labels[B,1]), masks).
    Ensures exactly 1 positive point per image.
    """
    B, C, H, W = masks.shape
    if C > 1:
        masks = masks[:, :1, :, :]

    coords_list = []
    labels_list = []
    new_mask_list = []

    for i in range(B):
        mask_i = masks[i, 0]  # [H, W]
        foreground_coords = torch.nonzero(mask_i)

        if foreground_coords.size(0) == 0:
            y = torch.randint(0, H, (1,))
            x = torch.randint(0, W, (1,))
            coords_list.append(torch.stack([y, x], dim=1))  # [1,2]
            labels_list.append(torch.tensor([1], dtype=torch.int32))
            new_mask_list.append(mask_i.unsqueeze(0))
        else:
            rnd_idx = torch.randint(0, foreground_coords.size(0), (1,))
            chosen = foreground_coords[rnd_idx]
            coords_list.append(chosen)  # [1,2]
            labels_list.append(torch.tensor([1], dtype=torch.int32))
            new_mask_list.append(mask_i.unsqueeze(0))

    pt_coords = torch.cat(coords_list, dim=0).float().to(imgs.device)  # [B,2]
    pt_coords = pt_coords.unsqueeze(1)  # [B,1,2]
    pt_labels = torch.cat(labels_list, dim=0).to(imgs.device)  # [B]
    pt_labels = pt_labels.unsqueeze(1)  # [B,1]
    new_masks = torch.stack(new_mask_list, dim=0).to(imgs.device)  # [B,1,H,W]

    return imgs, (pt_coords, pt_labels), new_masks


def generate_multi_point_prompt(imgs, masks, num_points=3):
    """
    Return (imgs, (pt_coords[B,num_points,2], pt_labels[B,num_points]), masks).
    Ensures exactly num_points positive points per image.
    """
    B, C, H, W = masks.shape
    if C > 1:
        masks = masks[:, :1, :, :]

    coords_list = []
    labels_list = []
    new_mask_list = []

    for i in range(B):
        mask_i = masks[i, 0]
        foreground_coords = torch.nonzero(mask_i)

        chosen_pts = []
        if foreground_coords.size(0) == 0:
            for _ in range(num_points):
                y = torch.randint(0, H, (1,))
                x = torch.randint(0, W, (1,))
                chosen_pts.append(torch.stack([y, x], dim=1))
        else:
            for _ in range(num_points):
                rnd_idx = torch.randint(0, foreground_coords.size(0), (1,))
                chosen = foreground_coords[rnd_idx]
                chosen_pts.append(chosen)

        chosen_pts = torch.cat(chosen_pts, dim=0)  # [num_points,2]
        coords_list.append(chosen_pts.unsqueeze(0))  # [1,num_points,2]

        lbl = torch.ones(num_points, dtype=torch.int32).unsqueeze(0)  # [1,num_points]
        labels_list.append(lbl)
        new_mask_list.append(mask_i.unsqueeze(0))

    pt_coords = torch.cat(coords_list, dim=0).float().to(imgs.device)  # [B,num_points,2]
    pt_labels = torch.cat(labels_list, dim=0).to(imgs.device)          # [B,num_points]
    new_masks = torch.stack(new_mask_list, dim=0).to(imgs.device)      # [B,1,H,W]

    return imgs, (pt_coords, pt_labels), new_masks


def generate_bbox_prompt(imgs, masks, ratio=0.75):
    """
    Return (imgs, bboxes[B,4], masks).
    Computes bbox around foreground and shrinks it by `ratio`.
    If mask empty => whole image.
    """
    B, C, H, W = masks.shape
    if C > 1:
        masks = masks[:, :1, :, :]

    all_bboxes = []
    for i in range(B):
        mask_i = masks[i, 0]
        coords = torch.nonzero(mask_i)
        if coords.size(0) == 0:
            bbox = [0, 0, W - 1, H - 1]
        else:
            y_min, x_min = coords.min(dim=0)[0]
            y_max, x_max = coords.max(dim=0)[0]
            center_x = (x_min + x_max).float() / 2.0
            center_y = (y_min + y_max).float() / 2.0

            width = (x_max - x_min).float() * ratio
            height = (y_max - y_min).float() * ratio

            new_xmin = torch.clamp((center_x - width / 2).round(), 0, W - 1)
            new_xmax = torch.clamp((center_x + width / 2).round(), 0, W - 1)
            new_ymin = torch.clamp((center_y - height / 2).round(), 0, H - 1)
            new_ymax = torch.clamp((center_y + height / 2).round(), 0, H - 1)

            bbox = [new_xmin.item(), new_ymin.item(), new_xmax.item(), new_ymax.item()]

        all_bboxes.append(bbox)

    bboxes_tensor = torch.tensor(all_bboxes, device=imgs.device, dtype=torch.float32)  # [B,4]
    return imgs, bboxes_tensor, masks


def generate_click_prompt(img, msk, pt_label=1):
    """
    (Deprecated) Prefer generate_single_point_prompt for exactly one point.
    """
    return generate_single_point_prompt(img, msk)


# ------------------ Evaluation Helper ------------------ #
def eval_seg(pred, true_mask, thresholds):
    """
    Evaluate (IoU, Dice, HD95) for single/multi-channel predictions.

    - c==1: returns (mean_iou, mean_dice, hd95)
    - c==2: returns (disc_iou, cup_iou, disc_dice, cup_dice)
    - c>2: returns (mean_iou, mean_dice)
    """
    b, c, h, w = pred.shape

    if c == 1:
        iou_sum, dice_sum = 0.0, 0.0
        for th in thresholds:
            vpred = (pred > th).float()
            vmask = (true_mask > th).float()
            iou_sum += iou(vpred, vmask)
            dice_sum += dice_coeff(vpred, vmask)
        mean_iou = iou_sum / len(thresholds)
        mean_dice = dice_sum / len(thresholds)
        hd95_val = hd95(pred, true_mask)
        return (mean_iou, mean_dice, hd95_val)

    if c == 2:
        disc_iou_sum, cup_iou_sum = 0.0, 0.0
        disc_dice_sum, cup_dice_sum = 0.0, 0.0
        for th in thresholds:
            disc_pred = (pred[:, 0] > th).float()
            disc_mask = (true_mask[:, 0] > th).float()
            cup_pred = (pred[:, 1] > th).float()
            cup_mask = (true_mask[:, 1] > th).float()

            disc_iou_sum += iou(disc_pred, disc_mask)
            cup_iou_sum += iou(cup_pred, cup_mask)
            disc_dice_sum += dice_coeff(disc_pred, disc_mask)
            cup_dice_sum += dice_coeff(cup_pred, cup_mask)

        n = float(len(thresholds))
        return (
            disc_iou_sum / n, cup_iou_sum / n,
            disc_dice_sum / n, cup_dice_sum / n
        )

    total_iou, total_dice = 0.0, 0.0
    for th in thresholds:
        vpred = (pred > th).float()
        vmask = (true_mask > th).float()
        iou_val = 0.0
        dice_val = 0.0
        for ch in range(c):
            iou_val += iou(vpred[:, ch], vmask[:, ch])
            dice_val += dice_coeff(vpred[:, ch], vmask[:, ch])
        total_iou += (iou_val / c)
        total_dice += (dice_val / c)

    mean_iou = total_iou / len(thresholds)
    mean_dice = total_dice / len(thresholds)
    return (mean_iou, mean_dice)
