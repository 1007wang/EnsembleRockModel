# Rock Image Classification

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.x-green.svg)

## Overview
This repository accompanies an SCI paper on rock image classification. We address long-tail and rare-class recognition across three major rock categories (igneous, metamorphic, sedimentary) and 49 sub-classes using a heterogeneous ensemble (InceptionV3 + EfficientNetB4). The framework introduces parameterized weight adaptation, branch gating, FPN-based hierarchical alignment, residual attention enhancement, attention-guided projection, and the ERA (Earth-Rock Attention) module to jointly capture fine-grained textures and global structures. On the 21,231-image RID dataset, the method achieves 92.45% accuracy (macro F1 = 0.9155, weighted F1 = 0.9213); minority classes (<150 samples, <0.7%) reach 92.08% average accuracy (80.00%–100.00% per class).

## Table of Contents
- [Environment & Dependencies](#environment--dependencies)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Advanced Usage (Systematic Comparisons)](#advanced-usage-systematic-comparisons)
- [Logs & Outputs](#logs--outputs)

## Environment & Dependencies
- Language/Platform: Python ≥ 3.9; Windows / Linux / macOS.
- DL stack: PyTorch ≥ 2.0 (recommend CUDA 11.x, e.g., 11.7+), torchvision ≥ 0.15.
- Core deps: numpy, Pillow, scikit-learn, matplotlib, seaborn, tqdm, opencv-python.
- Hardware guidance: ≥8GB GPU VRAM (16GB recommended); ≥16GB system RAM (32GB+ recommended); storage depends on data volume.
- Verified setup: Intel Xeon Silver 4216, 128GB RAM, NVIDIA RTX 2080 Ti (11GB), CUDA + Python, IDE: Visual Studio Code 1.105.1.

### Install
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
# For specific CUDA/Torch builds, follow https://pytorch.org/get-started/locally/
```

## Quick Start
- Prepare data under `processed_data/` with `class_mapping.json` (see structure below).
- Install dependencies (above), then train and test:
```bash
# Train ensemble (multistage scheduler)
python src/train_ensemble.py --data_dir processed_data --model_type ensemble --scheduler_type multistage

# Test all available checkpoints
python src/test_model.py --model all --checkpoint_dir checkpoints --test_data_dir test_data --output_dir test_reports
```
- Outputs: checkpoints in `checkpoints/`, reports in `test_reports/`, logs in `logs/`.

## Data Preparation
- Require `class_mapping.json` (class-path to index mapping).
- Organize data by class folders. Default train dir: `processed_data/`; default test dir: `test_data/`.
```
processed_data/
  IgneousRocks/
    andesite/*.jpg
    basalt/*.jpg
    ...
  MetamorphicRocks/
    marble/*.jpg
    ...
  SedimentaryRocks/
    limestone/*.jpg
    ...
  class_mapping.json
```
- Test set mirrors the same structure under `test_data/` (the provided `test_data/` can be used directly).

## Training
```bash
python src/train_ensemble.py \
  --data_dir processed_data \
  --model_type ensemble \  # {ensemble|resnet50|resnet50_optimized|efficientnet_b4|inceptionv3}
  --batch_size 16 \
  --num_epochs 200 \
  --learning_rate 1e-4 \
  --weight_decay 5e-4 \
  --scheduler_type multistage \  # {multistage|cosine}
  --patience 20 \
  --accumulation_steps 2
```
- Config file override: `--config configs/example.json` (JSON fields with the same names override defaults/CLI).
- Outputs: checkpoints in `checkpoints/`, training logs in `logs/`.

## Testing
```bash
python src/test_model.py \
  --model all \  # {ensemble|resnet_optimized|all}
  --checkpoint_dir checkpoints \
  --test_data_dir test_data \
  --output_dir test_reports \
  --batch_size 16
```
- Generates evaluation reports and visuals (e.g., confusion matrix) in `test_reports/`.

## Advanced Usage (Systematic Comparisons)
```bash
python src/run_comparison_experiments.py \
  --data_dir processed_data \
  --start_stage 1 --end_stage 5 \
  --basic_models resnet50 resnet50_optimized efficientnet_b4 inceptionv3 \
  --batch_size 32 \
  --num_epochs 100
```
- Useful flags: `--skip_stages` to skip stages; `--optimized_only` to run only optimized ResNet50; `--comparison_only` to compare original vs optimized ResNet50. The script generates configs and calls `src/train_ensemble.py`.

## Logs & Outputs
- Training: `logs/` (logs), `checkpoints/` (models), optional visuals in `visualization/`.
- Testing: `test_reports/`.
- Comparison experiments: `experiment_logs/` and generated `configs/`.

