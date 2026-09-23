# Pixel-level Mapping for Generalized AI-Generated Image Detection

This repository is a paper-based reproduction of **Beyond Semantic Features:
Pixel-level Mapping for Generalized AI-Generated Image Detection** (AAAI 2026,
[arXiv:2512.17350](https://arxiv.org/abs/2512.17350)). No official implementation
was linked by the paper or available when this reproduction was created.

The reproduced GenImage experiment trains a ResNet-50 on the official SD1.4 training
split and evaluates its fixed checkpoint on all eight official validation splits.
Images are transformed before classification using Equation 4:

```text
phi(v) = v - round(v / 256, 2) * 256,  v in {0, ..., 255}
```

The implementation uses a precomputed 256-value lookup table generated with NumPy so
that rounding semantics match the paper exactly. The same table is applied to every RGB
channel. The mapped values are passed directly to ResNet-50 without ImageNet
normalization.

## Reproduced protocol

The paper explicitly specifies:

- GenImage SD1.4 as training data and all GenImage generators for testing
- ResNet-50
- random 128x128 crops in training and center 128x128 crops at test time
- Adam, learning rate 2e-4, betas (0.9, 0.999), weight decay 2e-4
- batch size 128 and 200 epochs
- accuracy, average precision, and a fixed 0.5 fake-probability threshold

The paper does not state the model initialization, LR schedule, other augmentation,
random seed, validation/checkpoint-selection policy, or numerical precision. This
reproduction makes those choices explicit: scratch initialization, constant LR, no
unreported augmentation, seed 42, the complete official training split, final-epoch
selection, and BF16 on the local single GPU. Change only one field at a time for
ablation runs. The original paper used eight RTX 3090 GPUs; gradient accumulation here
preserves an effective batch size of 128 on one GPU but does not reproduce multi-GPU
BatchNorm statistics.

## Setup

```bash
cd /mnt/e/repos/PLM
uv sync --extra dev
```

The default dataset paths target the local GenImage installation under
`/mnt/f/datasets/GenImage`. Edit `configs/genimage_sd14_fixed.yaml` if needed.

## Validate the implementation

```bash
uv run pytest
```

One-batch GPU smoke test with a separate output directory:

```bash
bash scripts/train.sh \
  --epochs 1 \
  --max-batches 1 \
  --output outputs/smoke
```

## Train

```bash
bash scripts/train.sh
```

Resume at the next epoch after an interruption:

```bash
bash scripts/train.sh \
  --resume outputs/genimage_sd14_fixed/latest.pt
```

The run saves its exact config, sorted training manifest, corrupt-file log, per-epoch
metrics, `latest.pt`, ten-epoch snapshots, and `final.pt` under
`outputs/genimage_sd14_fixed/`.

## Evaluate all GenImage generators

```bash
bash scripts/evaluate_genimage.sh \
  --checkpoint outputs/genimage_sd14_fixed/final.pt
```

Evaluation writes per-generator real accuracy, fake accuracy, balanced/ordinary
accuracy, AP, and aggregate means to
`outputs/genimage_sd14_fixed/genimage_evaluation/`.

The paper reports fixed-mapping accuracies of 96.8, 98.9, 98.8, 98.7, 98.4, 98.2,
98.8, and 98.8 percent for Midjourney, SD1.4, SD1.5, ADM, GLIDE, Wukong, VQDM, and
BigGAN respectively, with 98.4 percent mean accuracy. These are reference values, not
results produced by this repository.

## Severity-controlled degradation pipeline

The input pipeline can apply deterministic degradations **before** the 128×128
crop and fixed pixel mapping. Levels range from `0` (exact clean input) to `5`
(extreme). Atomic profiles are `jpeg`, `webp`, `resize`, `blur`, `noise`,
`sharpen`, `color`, and `bit_depth`. Composed profiles are:

- `mixed`: a random subset of atomic operations whose size grows with severity.
- `sr`: a two-stage blind-super-resolution model with anisotropic blur, random
  resampling, Gaussian/Poisson noise, JPEG compression, and sinc ringing.
- `social`: resize, optional sharpening, and JPEG/WebP recompression.
- `screenshot`: fractional display scaling, recapture noise, and JPEG encoding.

Atomic severity endpoints are calibrated as follows; intermediate levels are
linearly spaced:

| Profile | Level 0 | Level 1 | Level 3 | Level 5 |
|---|---:|---:|---:|---:|
| JPEG quality | clean | 90 | 45 | 8 |
| WebP quality | clean | 90 | 40 | 5 |
| resize scale | clean | 0.8× | 0.4× | 0.125× |
| Gaussian blur radius | clean | 0.8 | 3.0 | 8.0 |
| Gaussian noise σ (0–255) | clean | 5 | 22 | 55 |
| retained bit depth | clean | 7 | 5 | 2 |

The composed profiles sample parameters from severity-dependent distributions,
so a particular level-4 image is not guaranteed to look worse than every level-3
image. Across a dataset, however, higher levels use stronger ranges and more
operations. Keep the seed fixed when comparing checkpoints.

Random parameters are derived from the path, seed, profile, and level, so every
image receives reproducible corruption regardless of worker count. Evaluate one
condition with:

```bash
bash scripts/evaluate_genimage.sh \
  --checkpoint outputs/genimage_sd14_fixed/epoch_020.pt \
  --degradation-profile sr --degradation-level 3 \
  --output outputs/genimage_sd14_fixed/degradation_sr_l3
```

Generate a six-level visual preview from any image with:

```bash
bash scripts/preview_degradations.sh path/to/image.jpg \
  --profile sr --output outputs/degradation_preview.jpg
```

### Random-degradation training

The 20-epoch robustness ablation samples an SR degradation level uniformly from
`0..5` for every training image. Level 0 retains clean examples. The random crop is
taken before degradation, following patch-based synthetic-degradation training used
in blind super-resolution.

The CPU version uses the PIL-based evaluation degradation exactly:

```bash
bash scripts/train.sh \
  --config configs/genimage_sd14_fixed_sr_random20.yaml
```

The faster CUDA version batches blur, resampling, Gaussian/Poisson noise, sinc
filtering, and an 8×8 DCT JPEG simulation on the GPU:

```bash
bash scripts/train.sh \
  --config configs/genimage_sd14_fixed_sr_random20_gpu.yaml
```

The CUDA JPEG transform reproduces JPEG's pixel-domain DCT and quantization effects
but does not perform entropy coding or chroma subsampling. Checkpoints record the
complete degradation configuration and per-epoch counts for all six levels.
