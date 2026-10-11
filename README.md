# CUB Fine-Grained Classification Reproduction

<<<<<<< HEAD
A compact PyTorch reproduction base for five FGVC methods:
ResNet50, ViT-B/16, WS-DAN, IELT, and P2P-Net. Each recipe has a three-seed training
entrypoint. The recorded P2P result currently contains one completed seed.


## Layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── THIRD_PARTY_NOTICES.md
├── configs/                  # One recipe per method, seeds 42/43/44
├── scripts/
│   ├── reproduce.py          # Local three-seed launcher
│   ├── train_elusive_cub.py   # ResNet50 and ViT training
│   ├── wsdan/                # WS-DAN trainer and numerical core
│   ├── ielt/                 # IELT trainer and model adapter
│   └── p2p/                  # P2P trainer and runtime adapter
├── third_party/              # Required upstream model code
└── results/
    └── cub.json              # Recorded aggregate and per-seed metrics
```

Source attribution and adaptation notes are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

=======
A compact PyTorch reproduction base for five bird classification methods:
ResNet50, ViT-B/16, WS-DAN, IELT, and P2P-Net. Each recipe has a three-seed training
entrypoint. The recorded P2P result currently contains one completed seed.

>>>>>>> d2ac2341f50a268b51ec30690936885a1b58331b
## Results

| Model | Train epochs | Seeds | Top-1 acc | Top-3 acc | Top-5 acc | Inference time |
|---|---:|---|---:|---:|---:|---:|
| ResNet50 | 50 | 42, 43, 44 | 84.96 ± 0.16% | 94.68% | 96.80% | 3.94 ms/image |
| ViT-B/16 | 50 | 42, 43, 44 | 86.08 ± 0.11% | 94.94% | 96.81% | 3.53 ms/image |
| WS-DAN / ResNet50 | 50 | 42, 43, 44 | 88.95 ± 0.09% | 96.30% | 97.80% | 7.69 ms/image |
| IELT / ViT-B/16 | 50 | 42, 43, 44 | 86.26 ± 0.25% | 95.38% | 97.22% | 7.22 ms/image |
| P2P-Net / ResNet50 | 50 | 42 only | 89.16% | 96.00% | 97.64% | 4.96 ms/image* |

Top-1 is the mean ± sample standard deviation across independent seeds, not an
ensemble. Top-3, Top-5 and latency columns show means. Exact per-seed values and
both timing scopes are in [results/cub.json](results/cub.json).

Measurements use an NVIDIA RTX5880-Ada-48Q, batch size 1, 20 warmup forwards,
100 timed forwards and CUDA synchronization. Model-only time excludes image
decoding, preprocessing and transfer. The first four methods use FP16; P2P uses
FP32, so these latencies are not a comparison at identical precision.

*P2P's 4.96 ms uses its logits-equivalent global-only forward. The full upstream
forward takes 12.07 ms/image. Training and accuracy evaluation use the full model;
the timing code checks logit equivalence before measuring the streamlined path.

## Protocol

- CUB-200-2011 official split: 5,994 training images and 5,794 test images, 200 classes.
- Full-image input at 448 × 448; no ground-truth bounding boxes or masks.
- 50 epochs, training batch size 16, seeds 42, 43 and 44.
- The final epoch is the reported checkpoint. The first four methods monitor the
  official test each epoch; P2P evaluates at epochs 1–4, every tenth epoch, and the
  final epoch. Test accuracy does not select checkpoints or trigger early stopping.
- ImageNet initialization: ResNet50 V2 for ResNet50/WS-DAN, Google ViT-B/16 augreg
  for ViT/IELT, and legacy ResNet50 V1 for P2P. No GBDD pretraining is used.

The first four methods follow the Elusive Images recipes. Their evaluation
transform is resize to 512 × 512 followed by a 448 × 448 center crop. P2P retains
its upstream resize550/crop448 transform, normalization, losses and cosine
schedule. Method-specific settings are documented in [configs/](configs/).
The `training_recipe` section documents the fixed numerical settings implemented
by each trainer; it is not a set of runtime overrides for optimizer/model design.

## Setup

Use Python 3.11 and a CUDA-compatible PyTorch installation. The recorded runs used
PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.15 and CUDA 12.8.

```bash
python -m pip install -r requirements.txt
```

Place the official extracted dataset at `data/CUB_200_2011/`, containing
`images/`, `classes.txt`, `images.txt`, `image_class_labels.txt`, and
`train_test_split.txt`. Supply these pretrained checkpoints under `weights/`:

| Methods | Local filename | Required checkpoint |
|---|---|---|
| ResNet50, WS-DAN | `weights/resnet50-11ad3fa6.pth` | [torchvision ImageNet-1K V2](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) |
| ViT, IELT | `weights/vit_b16_augreg_in1k.npz` | Google `B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz` |
| P2P | `weights/resnet50-19c8e357.pth` | [torchvision legacy ImageNet-1K V1](https://download.pytorch.org/models/resnet50-19c8e357.pth) |

The ViT checkpoint SHA256 must be
`69c9a2822a47838f57e522350b10ccc0d89d67fdcee07603e4ae3259b3350336`.
The launcher verifies each checkpoint's full SHA256 before training. Dataset images,
pretrained weights and generated runs are excluded by `.gitignore`.

## Reproduce

Run from the repository root. The launcher runs independent seeds sequentially
and writes each run to `runs/<model>/seed<seed>/`.

```bash
# One method, seeds 42/43/44
python scripts/reproduce.py --models wsdan

# All five methods, seeds 42/43/44
python scripts/reproduce.py --models resnet50 vit wsdan ielt p2p

# Print commands without starting training
python scripts/reproduce.py --models resnet50 vit --dry-run

# Resume runs created by this repository
python scripts/reproduce.py --models wsdan --resume
```

Training displays tqdm progress, saves checkpoints, evaluates the final model,
and measures model-only and preprocessing-inclusive latency. The launcher writes
an aggregate `runs/summary.json`; detailed outputs remain in each seed directory.
All dataset, weight and output paths are relative to this repository. Use a fresh
output directory when rerunning a completed experiment. Checkpoints produced by
the original development scripts use different source checksums; this package is
intended for fresh retraining, not resuming those older checkpoints.

<<<<<<< HEAD
## TO Investigate

| Model name | Train epoch | Top-1 acc | Top-3 acc | Top-5 acc | Inference time |
|---|---|---|---|---|---|
| [P2P-Net-RN50 (Paper)](https://arxiv.org/html/2203.15987) | 300 | 90.2% | — | — | 41 ms†（not sure） |
| [P2P-Net + LSDA (Paper)）](https://arxiv.org/html/2309.00399v1) | 300（As the paper's repository setting） | 91.0% | — | — | — |
| [I2-HOFI-RN50 (Paper)](https://d-nb.info/1352894165/34) | 150 | 90.12% | — | — | — |
| [PMG-RN50 (Paper)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf) | maximum 200 | 89.6% | — | — | 43 ms†） |
| [ECC + RN50 (Paper)）](https://arxiv.org/html/2407.04243v1) | 50 | 87.3% | — | — | — |
| [Relational Proxies-RN50 (Paper)](https://arxiv.org/html/2210.02149v1) | 200‡ | 92.00±0.01% | — | — | 130 ms/sample§ (Aircraft) |
| [TRD-RN50 (Paper)](https://papers.nips.cc/paper_files/paper/2023/file/beba7cfdac084a0f53f378d42cbe2824-Paper-Conference.pdf) | 1000‡ | 92.10±0.04% | — | — | 110 ms/sample§（Aircraft） |
| [INTS-Net-RN50 (Paper)](https://www.mdpi.com/2079-9292/12/7/1709) | 140 | 92.16±0.32% | — | — | 178 s/test set|
=======
## Layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── THIRD_PARTY_NOTICES.md
├── configs/                  # One recipe per method, seeds 42/43/44
├── scripts/
│   ├── reproduce.py          # Local three-seed launcher
│   ├── train_elusive_cub.py   # ResNet50 and ViT training
│   ├── wsdan/                # WS-DAN trainer and numerical core
│   ├── ielt/                 # IELT trainer and model adapter
│   └── p2p/                  # P2P trainer and runtime adapter
├── third_party/              # Required upstream model code
└── results/
    └── cub.json              # Recorded aggregate and per-seed metrics
```

Source attribution and adaptation notes are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
>>>>>>> d2ac2341f50a268b51ec30690936885a1b58331b
