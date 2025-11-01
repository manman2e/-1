# RCAN for Gaofen-2 Super-Resolution

This repository adapts [lornatang's RCAN implementation](https://github.com/lornatang/RCAN-PyTorch) and adds end-to-end tooling to train, evaluate, and deploy a Residual Channel Attention Network on Gaofen-2 satellite imagery.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data

Organize your Gaofen-2 dataset as paired low-resolution (LR) and high-resolution (HR) tiles:

```
/path/to/dataset/
  train/
    HR/
    LR/
  val/
    HR/
    LR/
```

If LR images are missing, the dataloader can downsample HR images on the fly using bicubic interpolation.

Update `configs/gaofen2.yaml` with the correct paths, scale factor, and training hyperparameters.

## Training

```bash
python scripts/train.py configs/gaofen2.yaml --devices 1 --accelerator gpu
```

This script uses PyTorch Lightning to handle checkpointing, logging, and mixed precision (configurable via the YAML file).

## Evaluation

```bash
python scripts/evaluate.py configs/gaofen2.yaml /path/to/checkpoint.ckpt
```

Computes the average PSNR on the validation set.

## Inference on Custom Imagery

```bash
python scripts/predict.py configs/gaofen2.yaml /path/to/checkpoint.ckpt /path/to/lr/images /path/to/output --tile-size 512 --overlap 32
```

Large images can be processed with tiling to avoid GPU memory issues. The script saves super-resolved outputs alongside the original filenames.

## Configuration

Key RCAN and training hyperparameters live in `configs/gaofen2.yaml`. Adjust network depth, patch size, batch size, and augmentation settings according to your GPU memory and dataset characteristics.
