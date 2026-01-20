# Ramtin Mojtahedi
#!/usr/bin/env python3
# ───────────────────────── reproducibility flags (set before torch) ──────────
import os, warnings, sys
# Tell HF Transformers to avoid importing torchvision (avoids nms/op registration)
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # optional, quieter

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # deterministic cuBLAS
os.environ["PYTHONHASHSEED"]          = "0"         # reproducible hashing
os.environ["TORCH_DETERMINISTIC"]     = "1"         # force deterministic kernels

# Safer CUDA allocator: *enforce* expandable segments even if CLI sets a partial value
# Safer CUDA allocator (avoid expandable_segments:True due to allocator assert on some stacks)
conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
parts = [p for p in conf.split(",") if p and not p.strip().startswith("expandable_segments")]
if not any(p.strip().startswith("max_split_size_mb:") for p in parts):
    parts.append("max_split_size_mb:192")
if not any(p.strip().startswith("garbage_collection_threshold:") for p in parts):
    parts.append("garbage_collection_threshold:0.6")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)

warnings.filterwarnings("ignore", message="Deterministic behavior was enabled")

# ─────────────────── EARLY pre‑parse of RoSA flags (before imports!) ─────────
def _preparse_rosa_flags_early():
    """
    Parse the new RoSA flags early and strip them from sys.argv BEFORE importing
    any module that might parse arguments (e.g., cfg). Values are stored in a
    module-global for re-injection after cfg.parse_args().
    """
    import argparse
    def _str2bool(v):
        if isinstance(v, bool): return v
        v = v.lower()
        if v in ("yes","true","t","y","1"):  return True
        if v in ("no","false","f","n","0"):  return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--rosa_ema_beta",        type=float, default=None)
    pre.add_argument("--rosa_ema_boost",       type=float, default=None)
    pre.add_argument("--rosa_safefloor_r",     type=int,   default=None)
    pre.add_argument("--rosa_safefloor_init",  type=float, default=None)
    pre.add_argument("--rosa_energy_balance",  type=_str2bool, nargs="?", const=True, default=None)
    pre.add_argument("--rosa_mix_init",        type=float, default=None)
    pre.add_argument("--rosa_mix_final",       type=float, default=None)
    pre.add_argument("--rosa_k_scale",         type=float, default=None)
    # NEW: expose these early too (optional; cfg will also parse them later)
    pre.add_argument("--rosa_k_train_init",    type=int,   default=None)
    pre.add_argument("--rosa_k_train_final",   type=int,   default=None)
    pre.add_argument("--rosa_k_warmup_frac",   type=float, default=None)
    pre.add_argument("--rosa_tau_init",        type=float, default=None)
    pre.add_argument("--rosa_tau_final",       type=float, default=None)
    pre.add_argument("--rosa_rankdrop_init",   type=float, default=None)
    pre.add_argument("--rosa_rankdrop_final",  type=float, default=None)
    pre.add_argument("--x_topk",               type=int,   default=None)
    pre.add_argument("--rosa_fc2_floor_train_init", type=float, default=0.12)
    pre.add_argument("--rosa_fc2_floor_train_final", type=float, default=0.25)
    pre.add_argument("--rosa_fc2_floor_eval",        type=float, default=0.33)


    known, remaining = pre.parse_known_args(sys.argv[1:])
    # Save known for later injection; strip all known flags from argv
    globals()["__ROSA_PREPARSED__"] = known
    sys.argv = [sys.argv[0]] + remaining

_preparse_rosa_flags_early()

# ───────────────────────── standard‑library imports ──────────────────────────
import json, time, copy, math, csv, inspect
from datetime import timedelta
from typing import Optional, Set

# ───────────────────────── third‑party imports ───────────────────────────────
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# CHG: enforce determinism on Ampere+ by disabling TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
# from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt

# Avoid hard deps on Transformers/PEFT for non‑LoRA runs (RoSA/Conv)
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None

# ────────────────────────────── project imports ──────────────────────────────
import cfg, function
from conf import settings
from dataset import get_dataloader
from utils import (
    create_logger, get_network, save_checkpoint, AttrDict,
    set_rosa_train_topk, l0_regularizer,
    build_adapter_aware_optimizer,   # stable optimizer for adapters
    schedule_rosa_from_args,         # ← NEW: use utils’ scheduler
    RoSAAdapter_Linear, RoSAAdapter_Conv2d,  # ← NEW: for logging helper
)

import utils as _utils

# ───────────────────── misc global tweaks / shims ────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)

# torch‑2.1 rename shim (GradScaler / autocast moved under torch.amp)
if hasattr(torch.cuda.amp, "GradScaler"):
    torch.cuda.amp.GradScaler = lambda *a, **k: torch.amp.GradScaler("cuda", *a, **k)
if hasattr(torch.cuda.amp, "autocast"):
    # Default to bfloat16 autocast unless callers override dtype
    def _autocast_bf16(*a, **k):
        k.setdefault("dtype", torch.bfloat16)
        return torch.amp.autocast("cuda", *a, **k)
    torch.cuda.amp.autocast = _autocast_bf16

# BCE / IoU patches for squeezed channel dim
_BCE = F.binary_cross_entropy_with_logits
F.binary_cross_entropy_with_logits = lambda x, y, *a, **k: _BCE(
    x, y.squeeze(-1) if y.dim() == x.dim() + 1 else y, *a, **k
)
_iou = _utils.iou
_utils.iou = lambda p, t: _iou(p, t.squeeze(-1) if t.ndim == p.ndim + 1 else t)

# ─────────────────── compact output switch ───────────────────────────────────
COMPACT_OUTPUT = True  # True → only final best-of summary is printed

def _print(*args, **kwargs):
    """Guarded print used throughout to keep notebook output short."""
    if not COMPACT_OUTPUT:
        print(*args, **kwargs)

def _dump_env_json(save_dir):
    import platform, json, sys, os
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "env": {k: os.environ.get(k) for k in [
            "CUDA_VISIBLE_DEVICES","CUBLAS_WORKSPACE_CONFIG","PYTHONHASHSEED",
            "PYTORCH_CUDA_ALLOC_CONF","OMP_NUM_THREADS","MKL_NUM_THREADS"
        ]},
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "env.json"), "w") as f:
        json.dump(info, f, indent=2)

def _log_disco_state_from_model(model: nn.Module, writer: SummaryWriter, epoch: int):
    try:
        import torch
        from utils import DiSCoAdapter_Linear, DiSCoAdapter_Conv2d  # type: ignore
        m0 = None
        for m in model.modules():
            if isinstance(m, (DiSCoAdapter_Linear, DiSCoAdapter_Conv2d)):
                m0 = m; break
        if m0 is None:
            return
        # rank r and eval-time top-k (0 => all)
        r   = int(getattr(m0, "r", 0))
        xtk = int(getattr(m0, "x_topk", torch.tensor(0)).item())
        writer.add_scalar("Train/DiSCo_rank_r",   r,   epoch)
        writer.add_scalar("Train/DiSCo_eval_topk", xtk, epoch)
    except Exception:
        pass

# ───────────────── adapter‑hyperparam probe ──────────────────────────────────
def adapter_hyperparams(model):
    """
    Report adapter rank/quant-bits if available.
    - PEFT (LoRA/QLoRA): from peft_config.
    - RoSA: infer r by scanning the first RoSA module (best-effort).
    - DiSCo: infer r by scanning the first DiSCo module (best-effort).
    """
    r = bits = None
    # PEFT path
    if getattr(model, "peft_config", None):
        cfg0 = next(iter(model.peft_config.values()))
        r = getattr(cfg0, "r", None)
        if hasattr(cfg0, "quantization_config"):
            qcfg = cfg0.quantization_config
            bits = 4 if getattr(qcfg, "load_in_4bit", False) else (
                   8 if getattr(qcfg, "load_in_8bit", False) else None)

    # RoSA path (best-effort)
    if r is None:
        try:
            from utils import RoSAAdapter_Linear, RoSAAdapter_Conv2d  # type: ignore
            for m in model.modules():
                if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
                    r = int(getattr(m, "r", None) or 0) or None
                    break
        except Exception:
            pass

    # DiSCo path (best-effort)
    if r is None:
        try:
            from utils import DiSCoAdapter_Linear, DiSCoAdapter_Conv2d  # type: ignore
            for m in model.modules():
                if isinstance(m, (DiSCoAdapter_Linear, DiSCoAdapter_Conv2d)):
                    r = int(getattr(m, "r", None) or 0) or None
                    break
        except Exception:
            pass

    return r, bits


# ─────────────── FLOP / param helpers + dup‑safe sum ─────────────────────────
def _to_gflops(x: Optional[float]) -> Optional[float]:
    if x is None: return None
    return float(x) if x < 1e6 else x / 1e9

def _to_mparams(x: Optional[int]) -> Optional[float]:
    if x is None: return None
    return float(x) if x < 1e5 else x / 1e6

def _unique_sum(params, *, trainable_only=False) -> int:
    seen: Set[int] = set(); total = 0
    for p in params:
        if trainable_only and not p.requires_grad: continue
        pid = id(p)
        if pid in seen: continue
        seen.add(pid); total += p.numel()
    return total

def _unwrap_dp(model_or_wrapper: nn.Module) -> nn.Module:
    if isinstance(model_or_wrapper, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model_or_wrapper.module
    return model_or_wrapper

def _replace_bnb_linear_for_ptflops(root: nn.Module) -> None:
    """
    Replace bitsandbytes Linear4bit/Linear8bitLt with shape-equivalent nn.Linear
    so ptflops can traverse the graph. Operates in-place on a *copied* model.
    """
    try:
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        Linear8 = Linear8bitLt  # alias if available
    except Exception:
        Linear4bit = None
        Linear8 = None

    def _recurse(mod: nn.Module):
        for name, child in list(mod.named_children()):
            hit4 = (Linear4bit is not None and isinstance(child, Linear4bit))
            hit8 = (Linear8 is not None and isinstance(child, Linear8))
            if hit4 or hit8:
                new = nn.Linear(child.in_features, child.out_features,
                                bias=(child.bias is not None))
                setattr(mod, name, new)
            else:
                _recurse(child)
    _recurse(root)

def model_stats(model, img_res=(3, 1024, 1024), gpu_id: int = 0):
    """
    Compute compute/size metrics with adapters included.

    IMPORTANT: To keep ptflops stable with RoSA's dynamic Top-K, we temporarily force
    all RoSA modules to dense eval (topk_eval=0) *only for profiling*. We then restore
    the original k for normal evaluation. This yields "dense" FLOPs; dynamic sparse FLOPs
    are not directly supported by ptflops.

    Returns:
        dict with flops, params, trainable, *_G/M fields, and 'rosa_eval_k' summary.
    """
    total_params     = _unique_sum(model.parameters())
    trainable_params = _unique_sum(model.parameters(), trainable_only=True)

    probe_dev = torch.device("cpu")
    m_eval = copy.deepcopy(_unwrap_dp(model)).eval()

    class _ImageEncoderOnly(nn.Module):
        def __init__(self, sam_model: nn.Module):
            super().__init__()
            self.enc = sam_model.image_encoder
        def forward(self, x):
            return self.enc(x)

    def _make_target(mod: nn.Module):
        if hasattr(mod, "image_encoder"):
            return _ImageEncoderOnly(mod), img_res, None
        return mod, img_res, None

    # (A) with adapters (profiling copy on CPU)
    m_with, in_res, _ = _make_target(m_eval)
    m_with = copy.deepcopy(m_with).to(probe_dev).eval()
    _replace_bnb_linear_for_ptflops(m_with)

    # --- NEW: snapshot + force RoSA dense Top‑K for ptflops ---
    def _snapshot_and_force_dense_topk(root: nn.Module):
        snaps = []
        for mod in root.modules():
            if hasattr(mod, "set_eval_topk") and hasattr(mod, "topk_eval"):
                try:
                    snaps.append((mod, int(getattr(mod, "topk_eval", 0))))
                    mod.set_eval_topk(0)  # dense for stable profiling
                except Exception:
                    pass
        return snaps

    def _restore_topk(snaps):
        for mod, k in snaps:
            try:
                mod.set_eval_topk(int(k))
            except Exception:
                pass

    # --- profile helper (unchanged) ---
    def _compute(m: Optional[nn.Module]) -> Optional[float]:
        if m is None:
            return None
        try:
            from ptflops import get_model_complexity_info  # noqa: WPS433
        except Exception as e:
            warnings.warn(f"ptflops not available or broken ({e}); skipping FLOPs.")
            return None
        try:
            macs, _ = get_model_complexity_info(
                m, in_res, as_strings=False, print_per_layer_stat=False, verbose=False
            )
            return macs * 2  # MACs → FLOPs
        except Exception as e:
            warnings.warn(f"ptflops compute failed ({e.__class__.__name__}: {e}); skipping FLOPs.")
            return None

    # Force dense top‑k → profile → restore
    snaps = _snapshot_and_force_dense_topk(m_with)
    flops_with_raw = _compute(m_with)
    _restore_topk(snaps)

    # (B) base-only (PEFT path) if available
    m_base = None
    if PeftModel and isinstance(m_eval, PeftModel):
        try:
            base = m_eval.get_base_model()
            m_base, _, _ = _make_target(base)
            m_base = copy.deepcopy(m_base).to(probe_dev).eval()
            _replace_bnb_linear_for_ptflops(m_base)
        except Exception:
            m_base = None

    flops_base_raw = _compute(m_base) if m_base is not None else None
    flops_extra_raw = (flops_with_raw - flops_base_raw) if (flops_with_raw is not None and flops_base_raw is not None) else None

    # Summaries in convenient units
    def _to_gflops(x: Optional[float]) -> Optional[float]:
        if x is None: return None
        return float(x) if x < 1e6 else x / 1e9

    def _to_mparams(x: Optional[int]) -> Optional[float]:
        if x is None: return None
        return float(x) if x < 1e5 else x / 1e6

    # Optional: summarize current RoSA eval Top‑K settings from the *original* model (not the CPU copy)
    rosa_eval_k = []
    try:
        from utils import RoSAAdapter_Linear, RoSAAdapter_Conv2d  # type: ignore
        for m in _unwrap_dp(model).modules():
            if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
                r  = int(getattr(m, "r", 0))
                ke = int(getattr(m, "topk_eval", 0))
                tag = str(getattr(m, "module_tag", "")) or "unknown"
                try:
                    k_use = m._scaled_k(ke) if hasattr(m, "_scaled_k") else min(ke, r)
                except Exception:
                    k_use = min(ke, r)
                rosa_eval_k.append({"module": tag, "r": r, "x_topk": ke, "k_use": int(k_use)})
    except Exception:
        pass

    return dict(
        flops       = flops_with_raw,
        params      = total_params,
        trainable   = trainable_params,
        flops_G     = _to_gflops(flops_with_raw),
        params_M    = _to_mparams(total_params),
        trainable_M = _to_mparams(trainable_params),
        flops_base_G          = _to_gflops(flops_base_raw),
        flops_adapter_extra_G = _to_gflops(flops_extra_raw),
        rosa_eval_k = rosa_eval_k,  # informational
    )

# ─────────────────────── training‑curve plots ────────────────────────────────
def save_plots(fig_dir, epochs, losses, dices):
    os.makedirs(fig_dir, exist_ok=True)
    def _plot(series, ylabel, fname):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(epochs, series)
        plt.xlabel("epoch"); plt.ylabel(ylabel); plt.title(ylabel)
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, fname)); plt.close()
    if losses: _plot(losses, "loss", "loss.png")
    if dices:  _plot(dices,  "Dice", "dice.png")

# ──────────────────────── serialization helpers ──────────────────────────────
def _to_serializable(obj):
    """Make nested dict/list/np types JSON‑serializable."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (torch.Tensor,)):
        return obj.detach().cpu().tolist()
    return obj

# ───────────────────── RoSA train‑time schedule (epoch‑wise) ─────────────────
def _log_rosa_state_from_model(model: nn.Module, writer: SummaryWriter, epoch: int):
    """
    Read the *actual* values applied inside the first RoSA module we find and
    log them to TensorBoard. This avoids drift between what we think we set
    and what the model is actually using.
    """
    try:
        m0 = None
        for m in model.modules():
            if isinstance(m, (RoSAAdapter_Linear, RoSAAdapter_Conv2d)):
                m0 = m
                break
        if m0 is None:
            return
        k_now = int(getattr(m0, "train_topk", 0))
        tau   = float(getattr(getattr(m0, "hard", None), "temperature", 0.0))
        p_rd  = float(getattr(m0, "rankdrop_p", 0.0))
        # sf_scale is a buffer tensor
        sf_s  = getattr(m0, "sf_scale", None)
        if torch.is_tensor(sf_s):
            sf_val = float(sf_s.detach().float().cpu().item())
        else:
            sf_val = float(sf_s) if sf_s is not None else 1.0

        writer.add_scalar("Train/RoSA_topk_k",  k_now, epoch)
        writer.add_scalar("Train/RoSA_tau",     tau,   epoch)
        writer.add_scalar("Train/RoSA_rankdrop", p_rd, epoch)
        writer.add_scalar("Train/RoSA_sf_scale", sf_val, epoch)
    except Exception:
        pass

# ───────────────────────────── train 1 fold ──────────────────────────────────
def train_one_fold(args, fold_idx: int, run_root: str, ds_name: str):
    """
    Trains one fold and returns a summary dict for CV aggregation.
    Prints a brief one-line Dice/IoU status after each validation.
    """
    # --- make per‑fold directories under adapter‑scoped run root -------------
    fold_dir = os.path.join(run_root, f"fold_{fold_idx:02d}")
    for sub in ("logs", "checkpoints", "results", "samples", "figures"):
        os.makedirs(os.path.join(fold_dir, sub), exist_ok=True)
    path_helper = {
        "log_path":     os.path.join(fold_dir, "logs"),
        "ckpt_path":    os.path.join(fold_dir, "checkpoints"),
        "results_path": os.path.join(fold_dir, "results"),
        "sample_path":  os.path.join(fold_dir, "samples"),
        "fig_path":     os.path.join(fold_dir, "figures"),
    }

    # clone args so outer loop remains clean and attach fold + paths
    args = copy.deepcopy(args)
    args.fold = int(getattr(args, "fold", fold_idx))
    args.path_helper = path_helper

    # reproducibility helper
    import random
    def set_seed(seed=0):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    set_seed(args.seed + args.fold)

    GPU = torch.device("cuda", args.gpu_device)

    # dataloaders
    g = torch.Generator(device="cpu").manual_seed(args.seed + args.fold)
    setattr(args, "num_workers", getattr(args, "num_workers", 0))
    train_loader, val_loader = get_dataloader(args, generator=g)

    # network
    net = get_network(
        args, args.net,
        use_gpu=True, gpu_device=GPU,
        distribution=args.distributed,
        steps=args.total_step,
        num_virtual_tokens=getattr(args, "num_virtual_tokens", None),
    )

    # work around HF‑Kwargs bug for PEFT/SAM
    from types import MethodType
    if PeftModel and isinstance(net, PeftModel):
        base = net.get_base_model()
        real_forward = base.forward
        def wrapped(self, *a, **kw):
            blacklist = {
                "input_ids","attention_mask","token_type_ids","position_ids",
                "labels","inputs_embeds",
                "output_attentions","output_hidden_states","return_dict",
            }
            kw = {k: v for k, v in kw.items() if k not in blacklist}
            return real_forward(*a, **kw)
        base.forward = MethodType(wrapped, base)

    # ─────────────── compute stats with FLOPs-safe fallback ───────────────────
    try:
        stats = model_stats(net, img_res=(3, args.image_size, args.image_size),
                            gpu_id=args.gpu_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release transient allocations from profiling
    except Exception as e:
        warnings.warn(f"ptflops failed ({e.__class__.__name__}: {e}). Falling back to parameter-only stats.")
        total_params     = _unique_sum(net.parameters())
        trainable_params = _unique_sum(net.parameters(), trainable_only=True)
        stats = dict(
            flops=None, flops_G=None, flops_base_G=None, flops_adapter_extra_G=None,
            params=total_params, trainable=trainable_params,
            params_M=total_params/1e6, trainable_M=trainable_params/1e6,
        )

    r, bits = adapter_hyperparams(net); stats.update(dict(rank=r, bits=bits))

    flopstr  = f"{stats['flops_G']:.2f} G" if stats.get('flops_G') else "N/A"
    flopbase = f"{stats['flops_base_G']:.2f} G" if stats.get("flops_base_G") else "N/A"
    flopextra= f"{stats['flops_adapter_extra_G']:.2f} G" if stats.get("flops_adapter_extra_G") else "N/A"
    logger  = create_logger(args.path_helper["log_path"])

    # Remove console logging to keep the terminal quiet
    import logging as _logging
    for h in list(getattr(logger, "handlers", [])):
        if isinstance(h, _logging.StreamHandler):
            logger.removeHandler(h)

    logger.info(
        f"[Fold {args.fold}] FLOPs total {flopstr}"
        + (f" | base {flopbase} | adapter +{flopextra}" if stats.get("flops_base_G") else "")
        + f" | Params {stats['params_M']:.2f} M | Trainable {stats['trainable_M']:.2f} M | r={r} bits={bits}"
    )
    _print(
        f"[Fold {args.fold}] FLOPs total {flopstr}"
        + (f" | base {flopbase} | adapter +{flopextra}" if stats.get("flops_base_G") else "")
        + f" | Params {stats['params_M']:.2f} M | Trainable {stats['trainable_M']:.2f} M | r={r} bits={bits}"
    )

    if args.pretrain:
        logger.info(f"Loading pretrained → {args.pretrain}")
        net.load_state_dict(torch.load(args.pretrain, map_location=GPU), strict=False)

    # ======= Optimizer (adapter-aware) =======================================
    optimizer = build_adapter_aware_optimizer(net, lr=args.lr, weight_decay=args.weight_decay)

    total_epochs   = args.epochs
    early_stop_pat = args.early_stop_threshold

    if args.scheduler == "cosine":
        warmup_steps = int(args.warmup_pct * total_epochs)
        def lr_lambda(ep):
            if ep < warmup_steps:
                return (ep + 1) / max(1, warmup_steps)
            progress = (ep - warmup_steps) / max(1, total_epochs - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    writer = SummaryWriter(args.path_helper["log_path"])

    if args.weights:
        ckpt = torch.load(args.weights, map_location=GPU)
        net.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info(f"Resumed from {args.weights} (epoch {ckpt.get('epoch','?')})")

    # Detect whether function.train_sam can accept an extra loss callable
    train_sam_sig = inspect.signature(function.train_sam)
    EXTRA_LOSS_PARAM = None
    for cand in ("extra_loss", "extra_loss_fn", "regularizer", "loss_hook"):
        if cand in train_sam_sig.parameters:
            EXTRA_LOSS_PARAM = cand
            break

    # ─────────────── training loop ────────────────────────────────────────────
    best_dice, best_hd95, last_improved = 0.0, None, 0
    losses, dices, ep_ids, epoch_results = [], [], [], []
    prompt_types = [p.strip() for p in args.prompt_types.split(",")]
    ref_ptype    = prompt_types[0]  # reference prompt for ckpt/early‑stop
    run_start    = time.time()

    # Save a copy of hparams for this fold
    with open(os.path.join(args.path_helper["results_path"], "hparams.json"), "w") as f:
        json.dump({k: _to_serializable(v) for k, v in vars(args).items()}, f, indent=2)

    for epoch in range(total_epochs):
        torch.cuda.reset_peak_memory_stats(GPU)
        ep_t0 = time.time()

        # ── RoSA train‑time schedule (use utils’ scheduler; epoch-wise pseudo-step)
        if args.adapter_type.lower() == "rosa":
            # map epoch → step in [0, total_epochs-1]  (avoid /0 when total_epochs==1)
            schedule_rosa_from_args(
                net,
                step=epoch,
                total_steps=max(1, total_epochs - 1),
                args=args
            )
            # log the *actual* values that the modules now hold
            _log_rosa_state_from_model(net, writer, epoch)
        
        # inside for epoch in range(total_epochs):
        if args.adapter_type.lower() == "disco":
            _log_disco_state_from_model(net, writer, epoch)

        # ── Optional L0 regularizer hook (keep 0.0 for fairness unless you opt-in)
        def _extra_loss_callable() -> torch.Tensor:
            if getattr(args, "rosa_l0", 0.0) > 0.0 and args.adapter_type.lower() == "rosa":
                return args.rosa_l0 * l0_regularizer(net)
            try:
                return torch.tensor(0.0, device=next(net.parameters()).device)
            except Exception:
                return torch.tensor(0.0)

        train_kwargs = {}
        if EXTRA_LOSS_PARAM is not None:
            train_kwargs[EXTRA_LOSS_PARAM] = _extra_loss_callable

        train_loss = function.train_sam(
            args, net, optimizer, train_loader,
            epoch, writer, vis=getattr(args, "vis", False),
            **train_kwargs
        )
        train_loss = float(train_loss.item() if torch.is_tensor(train_loss) else train_loss)
        writer.add_scalar("Train/Loss", train_loss, epoch)

        print(f"[Fold {args.fold}] Epoch {epoch:3d} | train-loss {train_loss:.4f} | Δ {time.time() - ep_t0:.1f}s")

        # ─────────────── validation & checkpoint (decoupled) ───────────────
        last_epoch = (epoch == total_epochs - 1)
        val_freq = int(getattr(args, "val_freq", args.save_freq))  # fallback to save_freq

        # Detect whether validation_sam supports a 'multimask_output' kwarg (safe-check)
        val_sig = inspect.signature(function.validation_sam)
        HAS_MMASK = ("multimask_output" in val_sig.parameters)

        do_val = (epoch % val_freq == 0) or last_epoch
        if do_val:
            net.eval()
            prompt_metrics = {}  # full-loader evaluation for each prompt type
            for ptype in prompt_types:
                v_kwargs = dict(prompt_type=ptype, visualize_prediction=False)
                # If supported, enable multi-proposal decoding when user sets >1
                if HAS_MMASK and int(getattr(args, "multimask_output", 1)) > 1:
                    v_kwargs["multimask_output"] = True

                v_loss, v_metrics = function.validation_sam(
                    args, val_loader, epoch, net, writer, **v_kwargs
                )
                v_loss = float(v_loss.item() if torch.is_tensor(v_loss) else v_loss)
                iou, dice, *rest = [float(x) for x in v_metrics]
                hd95 = rest[0] if rest else None
                prompt_metrics[ptype] = dict(loss=v_loss, iou=iou, dice=dice, hd95=hd95)

                # TB logs
                writer.add_scalar(f"Val/Loss/{ptype}", v_loss, epoch)
                writer.add_scalar(f"Val/IOU/{ptype}", iou, epoch)
                writer.add_scalar(f"Val/Dice/{ptype}", dice, epoch)
                if hd95 is not None:
                    writer.add_scalar(f"Val/HD95/{ptype}", hd95, epoch)

            # best prompt this epoch (for display only)
            best_prompt_epoch = max(prompt_metrics, key=lambda k: prompt_metrics[k]['dice'])
            best_dice_val_ep = prompt_metrics[best_prompt_epoch]['dice']
            _print(f"[Fold {args.fold}] → Best prompt this epoch: {best_prompt_epoch} (Dice={best_dice_val_ep:.4f})")

            # reference prompt for ckpt/early-stop
            cur_dice = prompt_metrics[ref_ptype]["dice"]
            cur_iou  = prompt_metrics[ref_ptype]["iou"]
            cur_hd95 = prompt_metrics[ref_ptype]["hd95"]

            # brief one-line console status
            will_improve = (cur_dice > best_dice)
            best_so_far  = cur_dice if will_improve else best_dice
            star = " ★" if will_improve else ""
            print(f"[E{epoch:03d}] Dice[{ref_ptype}]={cur_dice:.4f} | IoU={cur_iou:.4f} | best={best_so_far:.4f}{star}")

            # latency / RAM probe on one batch
            sample = next(iter(val_loader))
            torch.cuda.reset_peak_memory_stats(GPU)
            if sample["image"].dim() == 4:
                img_batch = sample["image"].to(GPU, non_blocking=True)
                batched = [{"image": img_batch[i], "original_size": sample.get("original_size", img_batch[i].shape[-2:])}
                        for i in range(img_batch.size(0))]
                denom = len(batched)
            else:
                img = sample["image"].to(GPU, non_blocking=True)
                batched = [{"image": img, "original_size": sample.get("original_size", img.shape[-2:])}]
                denom = 1
            t0 = time.time()
            with torch.no_grad():
                _ = net(batched_input=batched, multimask_output=False)
            latency_ms = (time.time() - t0) * 1e3 / denom
            inf_ram_gb = torch.cuda.max_memory_allocated(GPU) / (1024 ** 3)
            throughput_img_per_s = (1000.0 / latency_ms) if latency_ms > 0 else None

            # --- cleanup ---
            try: del _
            except: pass
            try: del batched
            except: pass
            try: del img_batch
            except: pass
            try: del img
            except: pass
            try: del sample
            except: pass
            torch.cuda.synchronize(GPU)
            torch.cuda.empty_cache()

            # paths
            best_ckpt = os.path.join(args.path_helper["ckpt_path"], f"best_{args.adapter_type}_fold{args.fold}.pth")
            last_ckpt = os.path.join(args.path_helper["ckpt_path"], f"last_fold{args.fold}.pth")

            # save policy:
            save_boundary = (epoch % args.save_freq == 0) or last_epoch
            if cur_dice > best_dice:
                best_dice, best_hd95, last_improved = cur_dice, cur_hd95, epoch
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": net.module.state_dict() if args.distributed != "none" else net.state_dict(),
                    "best_dice": best_dice,
                    "best_hd95": best_hd95,
                    "stats":     stats,
                    "adapter":   args.adapter_type,
                    "prompt_metrics": prompt_metrics,
                    "fold": args.fold,
                }, True, args.path_helper["ckpt_path"], filename=os.path.basename(best_ckpt))
                _print(f"[Fold {args.fold}] ★ new best Dice ({ref_ptype}) = {best_dice:.4f}")

            if save_boundary:
                save_checkpoint({
                    "epoch": epoch + 1,
                    "state_dict": net.module.state_dict() if args.distributed != "none" else net.state_dict(),
                }, False, args.path_helper["ckpt_path"], filename=os.path.basename(last_ckpt))

            # bookkeeping for final summary tables
            peak_mem = torch.cuda.max_memory_allocated(GPU) / (1024 ** 3)
            ep_time  = time.time() - ep_t0
            flops_G          = stats["flops_G"]
            params_M         = stats["params_M"]
            trainable_M      = stats["trainable_M"]
            trainable_pct    = 100.0 * stats["trainable"] / stats["params"]
            flops_per_dice_G = flops_G / cur_dice if flops_G else None
            params_per_dice  = params_M / max(cur_dice, 1e-8)

            epoch_results.append(dict(
                fold=args.fold, epoch=epoch, train_loss=train_loss, ref_ptype=ref_ptype,
                iou=cur_iou, dice=cur_dice, hd95=cur_hd95,
                epoch_time_sec=ep_time, flops_G=flops_G,
                flops_base_G=stats.get("flops_base_G"),
                flops_adapter_extra_G=stats.get("flops_adapter_extra_G"),
                params_M=params_M, trainable_M=trainable_M, trainable_pct=trainable_pct,
                flops_per_dice_G=flops_per_dice_G, params_per_dice_M=params_per_dice,
                peak_gpu_mem_gb=peak_mem, inf_latency_ms=latency_ms,
                inf_throughput_img_per_s=throughput_img_per_s, inf_ram_gb=inf_ram_gb,
                prompt_metrics=prompt_metrics, adapter=args.adapter_type, dataset=ds_name,
            ))

            losses.append(train_loss); dices.append(cur_dice); ep_ids.append(epoch)

            # early-stopping check
            if epoch - last_improved >= early_stop_pat:
                _print(f"[Fold {args.fold}] Early stop — no Dice improvement.")
                break
        scheduler.step()

    writer.close()

    # ---------- per‑fold artifacts ------------------------------------------------
    np_path = os.path.join(
        args.path_helper["results_path"],
        f"{args.exp_name}_{ds_name}_{args.adapter_type}_fold{args.fold}.npy"
    )
    json_path = os.path.join(
        args.path_helper["results_path"],
        f"{args.exp_name}_{ds_name}_{args.adapter_type}_fold{args.fold}_epochs.json"
    )
    with open(json_path, "w") as f:
        json.dump(_to_serializable(epoch_results), f, indent=2)
    np.save(np_path, epoch_results, allow_pickle=True)

    total_time = time.time() - run_start
    best_epoch = max(epoch_results, key=lambda x: x["dice"]) if epoch_results else {}

    meta = dict(
        fold                    = args.fold,
        adapter                 = args.adapter_type,
        dataset                 = ds_name,
        rank_r                  = stats["rank"],
        quant_bits              = stats["bits"],
        flops_G                 = stats["flops_G"],
        flops_base_G            = stats.get("flops_base_G"),
        flops_adapter_extra_G   = stats.get("flops_adapter_extra_G"),
        params_M                = stats["params_M"],
        trainable_M             = stats["trainable_M"],
        trainable_pct           = 100.0 * stats["trainable"] / stats["params"],
        best_dice               = best_epoch.get("dice", None),
        best_hd95               = best_epoch.get("hd95", None),
        best_iou                = best_epoch.get("iou", None),
        best_epoch              = best_epoch.get("epoch", None),
        ref_prompt              = best_epoch.get("ref_ptype", None),
        flops_per_dice_G        = best_epoch.get("flops_per_dice_G"),
        params_per_dice_M       = best_epoch.get("params_per_dice_M"),
        inf_latency_ms          = best_epoch.get("inf_latency_ms"),
        inf_throughput_img_per_s= best_epoch.get("inf_throughput_img_per_s"),
        inf_ram_gb              = best_epoch.get("inf_ram_gb"),
        peak_gpu_mem_gb         = best_epoch.get("peak_gpu_mem_gb"),
        total_training_time_sec = int(total_time),
        total_training_time     = str(timedelta(seconds=int(total_time))),
        epochs_run              = len(epoch_results),
        best_ckpt               = os.path.join(args.path_helper["ckpt_path"], f"best_{args.adapter_type}_fold{args.fold}.pth"),
        last_ckpt               = os.path.join(args.path_helper["ckpt_path"], f"last_fold{args.fold}.pth"),
    )
    meta_path = os.path.join(args.path_helper["results_path"], "meta.json")
    with open(meta_path, "w") as f:
        json.dump(_to_serializable(meta), f, indent=2)

    # save training curves
    save_plots(args.path_helper["fig_path"], ep_ids, losses, dices)

    # CSV: per‑prompt validation metrics per epoch
    per_prompt_csv = os.path.join(args.path_helper["results_path"], "per_prompt_metrics.csv")
    with open(per_prompt_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold","epoch","prompt_type","val_loss","val_iou","val_dice","val_hd95"])
        for er in epoch_results:
            ep = er["epoch"]
            for ptype, pm in er.get("prompt_metrics", {}).items():
                w.writerow([args.fold, ep, ptype, pm.get("loss"), pm.get("iou"), pm.get("dice"), pm.get("hd95")])

    # CSV: training log per epoch (plus ref prompt val snapshot)
    train_csv = os.path.join(args.path_helper["results_path"], "train_log.csv")
    with open(train_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "fold","epoch","train_loss","ref_prompt",
            "val_dice_ref","val_iou_ref","val_hd95_ref",
            "latency_ms","throughput_img_per_s","peak_gpu_mem_gb","inf_ram_gb","epoch_time_sec",
            "flops_G","flops_base_G","flops_adapter_extra_G","params_M","trainable_M","trainable_pct"
        ])
        for er in epoch_results:
            w.writerow([
                args.fold, er["epoch"], er["train_loss"], er["ref_ptype"],
                er["dice"], er["iou"], er["hd95"],
                er["inf_latency_ms"], er.get("inf_throughput_img_per_s"),
                er["peak_gpu_mem_gb"], er["inf_ram_gb"], er["epoch_time_sec"],
                er["flops_G"], er.get("flops_base_G"), er.get("flops_adapter_extra_G"),
                er["params_M"], er["trainable_M"], er["trainable_pct"],
            ])

    # ──────────────── FINAL COMPACT SUMMARY (fold) ────────────────────────────
    print(f"==== Fold {args.fold} Summary ====")
    if best_epoch:
        line = (f"Best Dice (ref '{best_epoch['ref_ptype']}'): {best_epoch['dice']:.4f} "
                f"(epoch {best_epoch['epoch']}) | IoU {best_epoch['iou']:.4f}")
        if best_epoch.get("hd95") is not None:
            line += f" | HD95 {best_epoch['hd95']:.2f}"
        if (stats.get("flops_base_G") is not None) and (stats.get("flops_G") is not None):
            line += (f" | FLOPs total {stats['flops_G']:.2f}G "
                     f"(base {stats['flops_base_G']:.2f}G + adapter {(stats.get('flops_adapter_extra_G') or 0.0):.2f}G)")
        elif stats.get("flops_G") is not None:
            line += f" | FLOPs total {stats['flops_G']:.2f}G"
        else:
            line += " | FLOPs total N/A"
        print(line)
    else:
        print("No validation epochs recorded.")
    print(f"Saved best checkpoint: {meta['best_ckpt']}")
    print(f"Results .npy: {np_path}")
    print(f"Meta JSON:    {meta_path}")
    print(f"Figures dir:  {args.path_helper['fig_path']}")
    print(f"Total time:   {meta['total_training_time']} | Epochs run: {meta['epochs_run']}")

    # return minimal fold summary for CV aggregation
    best_by_prompt = {}
    for p in prompt_types:
        best_p = None
        for er in epoch_results:
            m = er["prompt_metrics"].get(p)
            if m is None: continue
            if (best_p is None) or (m["dice"] > best_p["dice"]):
                best_p = dict(dice=m["dice"], iou=m["iou"], hd95=m.get("hd95"), epoch=er["epoch"])
        best_by_prompt[p] = best_p

    return dict(
        fold=args.fold,
        adapter=args.adapter_type,
        dataset=ds_name,
        ref_prompt=ref_ptype,
        best_ref_dice=best_epoch.get("dice", None),
        best_ref_iou=best_epoch.get("iou", None),
        best_ref_hd95=best_epoch.get("hd95", None),
        best_epoch=best_epoch.get("epoch", None),
        best_ckpt=meta["best_ckpt"],
        per_prompt_best=best_by_prompt,
        results_dir=args.path_helper["results_path"],
    )

# ─────────────────────────────── main() ──────────────────────────────────────
def main():
    # Parse project's main args
    args = cfg.parse_args()
    
    # Canonicalize common aliases for DiSCo (no effect on others)
    at = str(args.adapter_type).lower()
    if at in {"diso", "specadapter", "stadapter"}:
        args.adapter_type = "disco"

    # Inject early-preparsed RoSA flags (if user provided them)
    pre = globals().get("__ROSA_PREPARSED__", None)
    if pre is not None:
        for k, v in vars(pre).items():
            if v is not None:
                setattr(args, k, v)

    # ensure activation checkpointing is ON (applies to all adapters → fair)
    if not getattr(args, "grad_checkpointing", False):
        setattr(args, "grad_checkpointing", True)

    # reproducibility helper (base seed; folds offset by +fold)
    import random
    def set_seed(seed=0):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    set_seed(args.seed)

    args.save_freq  = getattr(args, "save_freq", 2)
    args.total_step = getattr(args, "total_step", None)

    BASE_DIR = "/mnt/largedrive1/rmojtahedi/medsam_adapter"
    ds_name  = os.path.basename(os.path.normpath(args.data_path)).lower()

    # New structure: experiments/Finaal_Run/<adapter_type>/<exp_ds_time>/
    run_root = os.path.join(
        BASE_DIR, "experiments", "Finaal_Run", args.adapter_type,
        f"{args.exp_name}_{ds_name}_{settings.TIME_NOW}"
    )
    os.makedirs(run_root, exist_ok=True)
    _dump_env_json(run_root)


    # CV only if user asked for it (n_folds > 1). Otherwise single fixed split.
    n_folds = max(1, int(getattr(args, "n_folds", 1)))
    do_cv   = (n_folds > 1)

    all_fold_summaries = []
    for k in range(n_folds):
        fold_summary = train_one_fold(args, fold_idx=k, run_root=run_root, ds_name=ds_name)
        all_fold_summaries.append(fold_summary)

    # ───────────── aggregate CV results across folds (if applicable) ──────────
    if do_cv:
        prompt_types = [p.strip() for p in args.prompt_types.split(",")]
        ref_prompt   = prompt_types[0]
        # collect per‑fold best ref Dice
        ref_dices = [fs["best_ref_dice"] for fs in all_fold_summaries if fs["best_ref_dice"] is not None]
        ref_iou   = [fs["best_ref_iou"]  for fs in all_fold_summaries if fs["best_ref_iou"]  is not None]
        ref_hd95  = [fs["best_ref_hd95"] for fs in all_fold_summaries if fs["best_ref_hd95"] is not None]

        # per‑prompt best Dice per fold → mean/std
        per_prompt = {p: [] for p in prompt_types}
        for fs in all_fold_summaries:
            bpp = fs.get("per_prompt_best", {})
            for p in prompt_types:
                m = bpp.get(p)
                if m and (m.get("dice") is not None):
                    per_prompt[p].append(m["dice"])

        agg = dict(
            adapter=args.adapter_type,
            dataset=ds_name,
            n_folds=n_folds,
            ref_prompt=ref_prompt,
            best_ref_dice_mean=(float(np.mean(ref_dices)) if ref_dices else None),
            best_ref_dice_std =(float(np.std(ref_dices, ddof=0)) if ref_dices else None),
            best_ref_iou_mean =(float(np.mean(ref_iou)) if ref_iou else None),
            best_ref_hd95_mean=(float(np.mean(ref_hd95)) if ref_hd95 else None),
            per_prompt_best_dice_mean={p: (float(np.mean(v)) if v else None) for p, v in per_prompt.items() },
            per_prompt_best_dice_std ={p: (float(np.std(v, ddof=0)) if v else None) for p, v in per_prompt.items() },
            folds=all_fold_summaries,
        )

        # save JSON + CSV
        cv_json = os.path.join(run_root, "cv_summary.json")  
        with open(cv_json, "w") as f:
            json.dump(agg, f, indent=2)

        cv_csv  = os.path.join(run_root, "cv_summary.csv")
        with open(cv_csv, "w", newline="") as f:
            w = csv.writer(f)
            header = ["fold","best_ref_dice","best_ref_iou","best_ref_hd95","best_epoch","best_ckpt"]
            w.writerow(header)
            for fs in all_fold_summaries:
                w.writerow([fs["fold"], fs["best_ref_dice"], fs["best_ref_iou"], fs["best_ref_hd95"], fs["best_epoch"], fs["best_ckpt"]])
            # add a blank line and write prompt means
            w.writerow([])
            w.writerow(["prompt","mean_best_dice","std_best_dice"])
            for p in prompt_types:
                w.writerow([p, agg["per_prompt_best_dice_mean"].get(p), agg["per_prompt_best_dice_std"].get(p)])

        print("==== Cross‑Validation Summary ====")
        print(f"Ref prompt '{ref_prompt}': Dice mean={agg['best_ref_dice_mean']}, std={agg['best_ref_dice_std']}")
        print(f"CV summary JSON: {cv_json}")
        print(f"CV summary CSV:  {cv_csv}")

    print("✔ training finished.")

if __name__ == "__main__":
    main()
