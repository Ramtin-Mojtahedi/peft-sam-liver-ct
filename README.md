# PEFT-SAM for Liver Tumor Segmentation in CT

Code accompanying **“Parameter-Efficient Fine-Tuning of Foundation Models for Liver Tumor Segmentation in CT”**.

This repository focuses on **parameter-efficient fine-tuning (PEFT)** of a SAM/MedSAM-style segmentation backbone for **CT liver tumor segmentation**, including prompt-driven evaluation (e.g., point and box prompts) and accuracy–efficiency benchmarking.

---

## What’s in this repo

- **PEFT adapters** evaluated in the paper (e.g., LoRA / QLoRA / convolutional adapters) and the proposed **DiSCo** adapter.
- **Prompt-driven segmentation** workflows (point/box prompting) aligned with interactive correction.
- **Reproducible training + validation** entry points and notebooks for quick experiments.

> **Note:** The dataset used in the paper is an in-house CRLM cohort and is **not included** here. The code is structured so you can plug in your own dataset using `dataset.json`.

---

## Repository layout (files)

