# DiSCo / PEFT Adapters for SAM on CT Liver Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Status](https://img.shields.io/badge/Status-Research%20Code-6c757d)
![License](https://img.shields.io/badge/License-Add%20LICENSE-lightgrey)

**GitHub “About” (copy/paste):**  
PEFT adapters (Conv-Adapter / LoRA / QLoRA / DiSCo) for SAM-style CT liver tumor segmentation with prompt-driven evaluation and accuracy–efficiency benchmarking.

---

## Paper

**Parameter-Efficient Fine-Tuning of Foundation Models for Liver Tumor Segmentation in CT**  
Ramtin Mojtahedi, Mohammad Hamghalam, Jacob J. Peoples, Richard K. G. Do, Amber L. Simpson

This repository provides code to **adapt a frozen SAM/MedSAM-style backbone** to CT liver tumor segmentation using **parameter-efficient fine-tuning (PEFT)**. Only lightweight adapter parameters are trained, enabling fast, low-memory adaptation and prompt-driven workflows.

> **Dataset note:** The in-house CRLM CT dataset used in the paper is **not included**.  
> You can run the same pipeline on your own dataset by configuring paths in `cfg.py`.

---

## What’s included

- PEFT training with a **frozen foundation segmentation backbone**
- Adapter implementations / wiring used in the study (**Conv-Adapter**, **LoRA**, **QLoRA**, **DiSCo**)
- Training + validation entry points
- One analysis notebook for tumor experiments

```mermaid
flowchart LR
  A[CT Images + Tumor Masks] --> B[Preprocess\nprecpt.py]
  B --> C[Train adapters\ntrain.py]
  C --> D[Validate / benchmark\nval.py]
  D --> E[Prompted inference\n(points/boxes)\n(config-driven)]
```

---

## Repository contents (matches current files)

> On Windows, Python extensions may be hidden (e.g., `cfg` → `cfg.py`).

```
.
├── Abdominal_Liver_sam_128.pth
├── cfg.py
├── function.py
├── MedSAM2D_Tumours.ipynb
├── precpt.py
├── train.py
├── utils.py
└── val.py
```

### File guide

- **`cfg.py`** — Central configuration (paths, hyperparameters, adapter selection, prompt/eval settings).
- **`function.py`** — Core model + training/evaluation logic (adapters, prompting utilities, loss/metrics wiring).
- **`utils.py`** — Shared utilities (I/O, transforms, logging, metrics helpers, timing, etc.).
- **`precpt.py`** — Preprocessing script to convert raw data into the format expected by training/evaluation.
- **`train.py`** — Training entry point (typical: freeze backbone → train only adapter parameters).
- **`val.py`** — Validation/evaluation entry point (segmentation metrics + optional compute metrics).
- **`MedSAM2D_Tumours.ipynb`** — Notebook for experiments, debugging, visualization, and analysis.
- **`Abdominal_Liver_sam_128.pth`** — Checkpoint/weights used in experiments (see checkpoint notes below).

---

## Setup (step-by-step)

### 1) Create a clean environment (recommended)

```bash
conda create -n disco-peft-sam python=3.10 -y
conda activate disco-peft-sam
pip install -U pip
```

### 2) Install PyTorch
Install a PyTorch build that matches your system (CPU or CUDA). Example:

```bash
pip install torch torchvision
```

(If you use CUDA, install the correct CUDA-enabled wheel for your driver/toolkit.)

### 3) Install common dependencies

```bash
pip install numpy scipy opencv-python scikit-image matplotlib tqdm pyyaml
```

---

## Data expectations

Because datasets differ, the most reliable reference is the dataset-loading logic inside **`utils.py` / `function.py`** and the path configuration in **`cfg.py`**.

In general, the pipeline expects **paired 2D images and binary tumor masks**, with consistent sizing.

A practical (recommended) layout:

```
data/
  images/   (CT slices)
  labels/   (binary masks aligned with images)
```

Common conventions:
- `images/` and `labels/` contain matching filenames (e.g., `0001.png` ↔ `0001.png`)
- masks are binary (0 background, 1 tumor) unless configured otherwise

If your loader expects a manifest (JSON/CSV), keep it next to the data and point to it in `cfg.py`.

---

## Preprocessing (step-by-step)

1) Open **`cfg.py`** and set:
- input data paths
- output directory for preprocessed slices/masks
- any preprocessing parameters used by your workflow

2) Run preprocessing:

```bash
python precpt.py
```

Typical CT → 2D preprocessing used in the study-style pipeline may include:
- intensity clipping (HU range) and scaling
- cropping to reduce search space (e.g., liver ROI when available)
- resizing/export to a fixed resolution suitable for SAM-style backbones

---

## Training (step-by-step)

1) Open **`cfg.py`** and configure:
- dataset paths
- output/log directory
- adapter type (Conv-Adapter / LoRA / QLoRA / DiSCo)
- training hyperparameters (epochs, lr, batch size, seed)
- prompt regime settings (if applicable)

2) Run training:

```bash
python train.py
```

Outputs depend on your config but typically include:
- adapter checkpoints
- training logs and validation summaries

---

## Validation / Evaluation (step-by-step)

Run evaluation:

```bash
python val.py
```

Typical evaluation reports:
- **Dice**
- **HD95**
- (optional) compute metrics such as latency / throughput / peak GPU memory, if enabled in your evaluation code

For fair benchmarking across adapters:
- keep the same input resolution
- keep batch size and device consistent
- keep prompt settings consistent

---

## Notebook workflow

Use the notebook for interactive experiments and visualization:

- **`MedSAM2D_Tumours.ipynb`**

Suggested checklist:
1) confirm dataset paths are correct
2) load a few samples and visualize image + mask
3) run a short train/val pass
4) visualize predictions (and prompts if used)

---

## Paper snapshot (optional summary table)

If you want a quick “results at a glance” section on the GitHub page, keep it short and update it to match your finalized runs. Example framing (paper-style interpretation):

| Adapter | Strength | Trade-off |
|---|---|---|
| Conv-Adapter | strong absolute accuracy | more trainable params than ultra-light methods |
| LoRA | strong accuracy, simple integration | can be heavier depending on rank / insertion sites |
| QLoRA | strong efficiency profile | quantization adds implementation complexity |
| DiSCo | excellent parameter efficiency | lower absolute accuracy in ultra-low parameter settings |

---

## Checkpoints (important for a clean public repo)

`Abdominal_Liver_sam_128.pth` is large enough that you generally **should not** commit it directly to git history for a public repo.

Recommended options:
- **Git LFS**
- **GitHub Releases**
- external storage + checksum (e.g., SHA256)

If the checkpoint is private or dataset-specific, keep it local and add it to `.gitignore`.

---

## Reproducibility tips

- Use **patient-level splits** if you have multiple slices per patient/case
- Fix and log:
  - random seed(s)
  - input resolution
  - adapter type + rank/scaling settings
  - prompt regime settings
  - device + batch size
- Keep evaluation conditions identical across adapters (same resolution, prompts, device)

---

## Citation

Update venue / DOI / arXiv once finalized:

```bibtex
@inproceedings{mojtahedi_peft_sam_liver_ct,
  title  = {Parameter-Efficient Fine-Tuning of Foundation Models for Liver Tumor Segmentation in CT},
  author = {Mojtahedi, Ramtin and Hamghalam, Mohammad and Peoples, Jacob J. and Do, Richard K. G. and Simpson, Amber L.},
  year   = {2026}
}
```

---

## Acknowledgments

This work was funded by National Institutes of Health and National Cancer Institute (R01CA233888, U01CA238444)

---

## Contact

For questions or issues, please open a GitHub Issue.  
Correspondence: **Ramtinrms@Gmail.com**

---
