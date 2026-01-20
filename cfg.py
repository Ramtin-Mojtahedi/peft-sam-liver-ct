# Ramtin Mojtahedi
# cfg.py
import argparse
import math


def _derive_rosa_defaults(args):
    """
    Fill safe, compute-lean defaults for RoSA when users don't provide schedule knobs.
    Works even when --total_step is None (static-K training).
    """
    if str(args.adapter_type).lower() != "rosa":
        return args

    # Effective rank r for RoSA (mid_dim if provided, else a conservative default = 32)
    r = int(args.mid_dim) if args.mid_dim is not None else 32
    r = max(1, r)

    # Helper to clamp integer K into [0, r]
    def _clamp_k(v):
        if v is None:
            return None
        v = int(v)
        if v < 0:
            v = 0
        return min(v, r)

    # If x_topk not set but x_topk_frac is provided, derive x_topk from fraction of r
    if (getattr(args, "x_topk", None) in (None, 0)) and (args.x_topk_frac is not None):
        try:
            frac = float(args.x_topk_frac)
            if frac > 0.0:
                args.x_topk = max(0, int(math.ceil(frac * r)))
        except Exception:
            pass

    # Clamp user-provided values (if any)
    args.x_topk = _clamp_k(args.x_topk)
    args.rosa_k_train_init = _clamp_k(args.rosa_k_train_init)
    args.rosa_k_train_final = _clamp_k(args.rosa_k_train_final)

    # If neither train K is provided, choose sensible defaults
    if args.rosa_k_train_init is None and args.rosa_k_train_final is None:
        if args.total_step is None:
            # No total steps => static compute-lean training by default
            k_static = min(32, r)
            # If user set an eval x_topk, ensure train_final >= eval K (keeps eval <= train)
            if args.x_topk and args.x_topk > 0:
                k_static = max(k_static, int(args.x_topk))
            args.rosa_k_train_init = k_static
            args.rosa_k_train_final = k_static
        else:
            # With total steps available => do dense -> sparse by default
            args.rosa_k_train_init = 0
            args.rosa_k_train_final = int(args.x_topk) if (args.x_topk and args.x_topk > 0) else min(32, r)

    # If only one side is provided, fill the other
    if args.rosa_k_train_init is None and args.rosa_k_train_final is not None:
        args.rosa_k_train_init = int(args.rosa_k_train_final)
    if args.rosa_k_train_final is None and args.rosa_k_train_init is not None:
        # Prefer to end at max(eval K, init)
        end_k = int(args.rosa_k_train_init)
        if args.x_topk and args.x_topk > 0:
            end_k = max(end_k, int(args.x_topk))
        args.rosa_k_train_final = _clamp_k(end_k)

    # Ensure eval K does not exceed rank r
    args.x_topk = _clamp_k(args.x_topk)

    # Ensure final train K is at least eval K (so eval isn't "denser" than train)
    if args.x_topk and args.x_topk > 0:
        args.rosa_k_train_final = _clamp_k(max(int(args.rosa_k_train_final), int(args.x_topk)))

    # Clamp warmup fraction
    args.rosa_k_warmup_frac = float(min(max(args.rosa_k_warmup_frac, 0.0), 0.95))

    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configuration for SAM-adapter / (Q)LoRA / RoSA / DiSCo training"
    )

    # ── Core experiment setup ────────────────────────────────────────────────
    parser.add_argument('-net',            type=str,   default='sam',
                        help='Network type')
    parser.add_argument('-baseline',       type=str,   default='unet',
                        help='Baseline network type')
    parser.add_argument('-encoder',        type=str,   default='default',
                        help='Encoder type (SAM backbone variant)')
    parser.add_argument('-seg_net',        type=str,   default='transunet',
                        help='Segmentation network type')
    parser.add_argument('-mod',            type=str,   default='sam_adpt',
                        help='Model mode: seg | cls | val_ad')
    parser.add_argument('-exp_name',       type=str,   default='msa_test_isic',
                        help='Experiment name')
    parser.add_argument('-dataset',        type=str,   default='isic',
                        help='Dataset name')
    parser.add_argument('-type',           type=str,   default='map',
                        help='Condition type: ave | rand | rand_map')
    parser.add_argument('-vis',            type=int,   default=None,
                        help='Visualization flag')
    parser.add_argument('-reverse',        action='store_true',
                        help='Enable adversarial reverse')

    # ── Pre-training & checkpoints ───────────────────────────────────────────
    parser.add_argument('-pretrain',       type=str,   default=None,
                        help='Path to pretrained model (weights loaded with strict=False)')
    parser.add_argument('-base_weights',   type=str,   default=0,
                        help='Baseline weights file')
    parser.add_argument('-sim_weights',    type=str,   default=0,
                        help='Sim weights file')
    parser.add_argument('-weights',        type=str,   default=0,
                        help='Weights file to resume')
    parser.add_argument('-c', '--sam_ckpt', type=str,
                        default='sam_vit_b_01ec64.pth',
                        help='Absolute path to SAM checkpoint (.pth)')

    # ── Training schedule ────────────────────────────────────────────────────
    parser.add_argument('-epochs',         type=int,   default=20,
                        help='Total training epochs')
    parser.add_argument('-epoch_ini',      type=int,   default=1,
                        help='Initial epoch (for resume)')
    parser.add_argument('-early_stop_threshold', type=int, default=5,
                        help='Epochs with no improvement before early stop')
    parser.add_argument('-save_freq',      type=int,   default=2,
                        help='Checkpoint every N epochs')
    parser.add_argument('-val_freq',       type=int,   default=5,
                        help='Interval between validations (in epochs)')
    parser.add_argument('-warm',           type=int,   default=1,
                        help='Warm-up phase (epochs)')
    parser.add_argument('-total_step',     type=int,   default=None,
                        help='Total steps hint for some schedulers (auto if None)')

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    parser.add_argument('-lr',             type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('-imp_lr',         type=float, default=3e-4,
                        help='Implicit learning rate')
    parser.add_argument('-weight_decay',   type=float, default=0.01,
                        help='AdamW weight-decay (0.0 disables)')
    parser.add_argument('-beta2',          type=float, default=0.999,
                        help='Second β for Adam/AdamW')
    parser.add_argument('-scheduler',      type=str,   default='cosine',
                        choices=['step', 'cosine'],
                        help='LR scheduler type')
    parser.add_argument('-warmup_pct',     type=float, default=0.05,
                        help='Warm-up fraction (cosine)')
    parser.add_argument('--lr_scaled',     action='store_true',
                        help='Disable automatic LR-halving for QLoRA')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Clip gradient L2-norm (0 disables)')
    parser.add_argument('-grad_checkpointing', action='store_true',
                        help='Enable gradient checkpointing')

    # ── PEFT / adapter options ───────────────────────────────────────────────
    parser.add_argument(
        '-adapter_type',
        type=str,
        default='lora',
        choices=['lora', 'qlora', 'conv', 'rosa', 'full', 'disco', 'diso', 'specadapter', 'stadapter'],
        help='Adapter configuration ("full" = full fine-tune; supported: lora, qlora, conv, rosa, disco).',
    )
    parser.add_argument('-qlora_lowperf', action='store_true',
                        help='Use FP4 quantizer + higher-rank LoRA (low-perf QLoRA)')
    parser.add_argument('-mid_dim', type=int, default=None,
                        help='Explicit rank / mid dimension (overrides defaults)')
    parser.add_argument('--num_virtual_tokens', type=int, default=None,
                        help='Number of virtual tokens for prefix/prompt tuning (if applicable)')

    # Targeting which modules to adapt
    parser.add_argument(
        '--target_modules',
        type=str,
        default=None,
        help=('Comma-separated module name suffixes to adapt. '
              'Example: "attn.qkv,attn.proj,mlp.fc1,mlp.fc2". '
              'If the value starts with "(", it is treated as a regex applied to full module names.')
    )

    # RoSA / DiSCo spectral top-k control
    parser.add_argument('--x_topk', type=int, default=0,
                        help='RoSA/DiSCo: eval-time hard Top-K spectral modes per layer (0 = use all r). '
                             'Train-time K is scheduled separately via --rosa_k_*.')
    parser.add_argument('--x_topk_frac', type=float, default=None,
                        help='RoSA/DiSCo: eval-time Top-K as a fraction of rank r (e.g., 0.06). '
                             'If both --x_topk and --x_topk_frac are set, --x_topk takes precedence.')

    # Extra knobs used by custom adapters in utils.py
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--share_st',    dest='share_st', action='store_true',
                    help='Share ST conv across blocks (kept for back-compat)')
    group.add_argument('--no_share_st', dest='share_st', action='store_false',
                    help='Do not share ST conv across blocks')
    parser.set_defaults(share_st=True)

    parser.add_argument('--conv_kernel', type=int, default=3,
                        help='Kernel size for ConvAdapter depthwise convs')
    parser.add_argument('--patch_embed_adapter', type=str, default='none',
                        choices=['none', 'conv'],
                        help='If "conv", wrap ViT patch_embed.proj (Conv2d) with our ConvAdapter in addition to PEFT adapters')

    # ── RoSA-specific sparsity & schedule ────────────────────────────────────
    parser.add_argument('--rosa_dropout', type=float, default=None,
                    help='Override RoSA spectral dropout (default 0.04).')
    parser.add_argument('--rosa_l0', type=float, default=0.0, help='RoSA: L0 (hard-concrete) regularization strength for gates (e.g., 1e-5). 0 disables.')
    parser.add_argument('--rosa_k_train_init', type=int, default=None, help='RoSA: initial train-time top-K (default auto: 0 if total_step set; else min(32, r)).')
    parser.add_argument('--rosa_k_train_final', type=int, default=None, help='RoSA: final train-time top-K (default auto: max(x_topk, min(32, r))).')
    parser.add_argument('--rosa_k_warmup_frac', type=float, default=0.4, help='RoSA: fraction of total steps to decay K from init to final (linear). '
                        'Ignored if total_step is None (static training).')
    parser.add_argument('--rosa_temp_init', type=float, default=2.0/3.0, help='RoSA: initial hard-concrete temperature (default 2/3).')
    parser.add_argument('--rosa_temp_final', type=float, default=2.0/3.0, help='RoSA: final hard-concrete temperature (default 2/3).')
    parser.add_argument('--rosa_score_temp', type=float, default=1.0, help='RoSA: score sharpening exponent for rank scores (>=0.5). 1.0 = no change.')
    parser.add_argument('--rosa_mix_init', type=float, default=None, help='RoSA: teacher/dense mix coefficient at start of training (override).')
    parser.add_argument('--rosa_mix_final', type=float, default=None, help='RoSA: teacher/dense mix coefficient at end of training (override).')
    parser.add_argument('--rosa_p_keep_init', type=float, default=0.95,
                        help='RoSA: initial keep probability for HardConcrete gates (suggest 0.90–0.93).')
    parser.add_argument('--rosa_k_scale', type=float, default=None, help='RoSA: base per-module k_scale multiplier (default 1.05 if unset).')

    # Aliases for hard-concrete temperature (train.py/utils.py use tau names)
    parser.add_argument('--rosa_tau_init', type=float, default=None, help='RoSA: hard-concrete temperature init (alias). If None, uses --rosa_temp_init.')
    parser.add_argument('--rosa_tau_final', type=float, default=None, help='RoSA: hard-concrete temperature final (alias). If None, uses --rosa_temp_final.')

    # ── Data loading & model dims ────────────────────────────────────────────
    parser.add_argument('-image_size',     type=int,   default=256,
                        help='Input image size')
    parser.add_argument('-out_size',       type=int,   default=256,
                        help='Output size')
    parser.add_argument('-patch_size',     type=int,   default=2,
                        help='Patch size')
    parser.add_argument('-dim',            type=int,   default=512,
                        help='Transformer dimension')
    parser.add_argument('-depth',          type=int,   default=1,
                        help='Transformer depth')
    parser.add_argument('-heads',          type=int,   default=16,
                        help='Attention heads')
    parser.add_argument('-mlp_dim',        type=int,   default=1024,
                        help='MLP dimension')
    parser.add_argument('-uinch',          type=int,   default=1,
                        help='UNet input channels')

    parser.add_argument('-w',              type=int,   default=4,
                        help='Dataloader workers')
    parser.add_argument('-b',              type=int,   default=2,
                        help='Batch size')
    parser.add_argument('-s',              action='store_true',
                        help='Shuffle dataset')
    parser.add_argument('-data_path',      type=str,   default='../data',
                        help='Root path to segmentation data')

    # ── 3-D / RoI cropping ──────────────────────────────────────────────────
    parser.add_argument('-thd',            action='store_true',
                        help='Enable 3-D processing')
    parser.add_argument('-chunk',          type=int,   default=None,
                        help='Crop volume depth')
    parser.add_argument('-roi_size',       type=int,   default=96,
                        help='RoI resolution')
    parser.add_argument('-evl_chunk',      type=int,   default=None,
                        help='Evaluation chunk size')

    # ── Prompt evaluation & sampling ────────────────────────────────────────
    parser.add_argument('--prompt_types', type=str,
                        default='single_point,multi_point,bbox_075,bbox_05',
                        help='Comma-separated prompt types to evaluate')
    parser.add_argument('-num_sample',     type=int,   default=4,
                        help='Positive + negative samples per image')
    parser.add_argument('-multimask_output', type=int, default=1,
                        help='Masks per image for multi-class segmentation')

    # ── Hardware & reproducibility ──────────────────────────────────────────
    parser.add_argument('-gpu',            action='store_true',
                        help='Use GPU')
    parser.add_argument('-gpu_device',     type=int,   default=0,
                        help='GPU device id')
    parser.add_argument('-sim_gpu',        type=int,   default=0,
                        help='Split sim to this GPU')
    parser.add_argument('-distributed',    type=str,   default='none',
                        help='Comma-separated GPU ids for DDP')
    parser.add_argument('-seed',           type=int,   default=42,
                        help='Global random seed')

    # ── Cross-validation (used by Liver dataset) ────────────────────────────
    parser.add_argument('-n_folds',        type=int,   default=5,
                        help='Number of folds for K-fold CV (Liver dataset)')
    parser.add_argument('-fold',           type=int,   default=0,
                        help='Which fold to run (0-based). When running all folds, this is set internally.')
    parser.add_argument('-cv_seed',        type=int,   default=42,
                        help='Seed for patient-level fold shuffling (defaults to --seed if you keep 42 and also change --seed).')

    # Regularize (legacy; safe to keep)
    parser.add_argument('--lambda_gate',   type=float, default=0.0,
                        help='L1 penalty on .gate params (use 0.0 for fair comparisons)')

    # ----- parse -------------------------------------------------------------
    args = parser.parse_args()

    # Keep utils.schedule_rosa_from_args happy if only the "temp" flags are set
    if args.rosa_tau_init is None:
        args.rosa_tau_init = args.rosa_temp_init
    if args.rosa_tau_final is None:
        args.rosa_tau_final = args.rosa_temp_final

    # Post-process RoSA defaults so it is stable and compute-lean without total_step.
    args = _derive_rosa_defaults(args)

    return args
