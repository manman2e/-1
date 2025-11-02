# RCAN for Gaofen-2 Super-Resolution

This repository adapts [lornatang's RCAN implementation](https://github.com/lornatang/RCAN-PyTorch) and adds end-to-end tooling to train, evaluate, and deploy a Residual Channel Attention Network on Gaofen-2 satellite imagery. The pipeline now reads four-band RGB+NIR GeoTIFF tiles (and other multi-band rasters), letting the model consume the full spectral stack without manual preprocessing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data

Organize your Gaofen-2 dataset as paired low-resolution (LR) and high-resolution (HR) tiles. Each tile should contain the four Gaofen-2 bands (R, G, B, and NIR) stored in `.tif`/`.tiff` containers or another raster format that exposes at least four channels:

```
/path/to/dataset/
  train/
    HR/
    LR/
  val/
    HR/
    LR/
```

If LR images are missing, the dataloader can downsample HR images on the fly using bicubic interpolation while preserving all spectral channels. You can also pre-generate an LR folder from your HR Gaofen-2 tiles with:

```bash
python scripts/generate_lr.py /data/gaofen2/train/HR /data/gaofen2/train/LR --scale 4
```

The script keeps all four spectral bands intact, handles GeoTIFF metadata, and defaults to bicubic interpolation. Pass `--suffix _LR` if you prefer to append a suffix to each filename instead of mirroring the HR names.

Update `configs/gaofen2.yaml` with the correct paths, scale factor, number of channels (defaults to 4), and training hyperparameters.

## Training

```bash
python scripts/train.py configs/gaofen2.yaml --devices 1 --accelerator gpu \
    --train-hr /data/gaofen2/train/HR --train-lr /data/gaofen2/train/LR \
    --val-hr /data/gaofen2/val/HR --val-lr /data/gaofen2/val/LR
```

CLI flags let you override dataset paths, batch size, patch size, and training epochs without editing the YAML file. This script
uses PyTorch Lightning to handle checkpointing, logging, and mixed precision (configurable via the YAML file or CLI overrides).

## Evaluation

```bash
python scripts/evaluate.py configs/gaofen2.yaml /path/to/checkpoint.ckpt \
    --val-hr /data/gaofen2/val/HR --val-lr /data/gaofen2/val/LR
```

Computes the average PSNR on the validation set. Pass `--split train` to score the training split instead.

## Inference on Custom Imagery

```bash
python scripts/predict.py /path/to/checkpoint.ckpt /path/to/lr/images /path/to/output \
    --config configs/gaofen2.yaml --tile-size 512 --overlap 32
```

Large images can be processed with tiling to avoid GPU memory issues. GeoTIFF metadata such as channel count and data type are preserved when saving predictions so RGB+NIR products can be ingested directly into downstream pipelines. The `--config` flag is optional because checkpoints already store the training hyperparameters; supply it only when you want to override them.

## Configuration

Key RCAN and training hyperparameters live in `configs/gaofen2.yaml`. Adjust network depth, patch size, batch size, and augmentation settings according to your GPU memory and dataset characteristics.
