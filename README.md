# Knowledge distillation on CIFAR-100: ResNet teacher, ResNet and ViT/DeiT students

This repository studies knowledge distillation on CIFAR-100 using ResNet and Vision Transformer
models implemented from scratch in PyTorch. It compares plain cross-entropy training, Hinton-style
soft distillation and DeiT-style distillation with a dedicated distillation token, with the aim of
understanding what a teacher actually contributes to a student, in particular to a ViT trained on
a small dataset. The goal is understanding rather than state-of-the-art accuracy.

The work is the first step of a longer path towards BEVFormer:
KD → ResNet → ViT → DeiT → DETR → Deformable DETR → camera geometry / BEV (LSS) → BEVFormer.

## Repository layout

| Path | Contents |
|---|---|
| `src/kd/resnet.py` | Block-based ResNet: `Residual`, `resBlock`, `Resnet(in_channels, num_classes, width, blocks)` |
| `src/kd/vit.py` | ViT (`Attention`, `vitBlock`, `ViT`) with an optional distillation token (`distill=True`) |
| `src/kd/c100.py` | Data loading, losses, training loop, experiment definitions |
| `src/kd/soft_dst.py` | Additional experiment: ViT with distillation token and soft distillation |
| `build_kaggle.py` | Merges the modules into single files that can be pasted into one Kaggle cell |
| `kaggle_c100.py`, `kaggle_soft_dst.py` | Generated files (do not edit by hand) |
| `results/*.json` | Per-epoch test accuracy, parameter count and training time for every run |

## Models

**ResNet.** The stem is a 3×3 convolution with stride 1 and no max-pooling (the images are 32×32),
followed by four stages of two blocks each (`blocks=(2,2,2,2)` gives ResNet-18; other tuples give
other depths). Stage `k` has `width · 2^k` channels. A block computes `act(main(x) + shortcut(x))`;
the shortcut is a 1×1 convolution with batch normalisation when the stride or the number of
channels changes, and the identity otherwise. The activation is GELU. The head is adaptive average
pooling, flatten and a linear layer. The teacher is `Resnet(width=64)` with 11.2M parameters; the
students have width 16 (0.7M) and width 32 (2.8M).

**ViT.** 4×4 patches (a convolution with kernel = stride = 4) give 64 tokens plus a `[CLS]` token,
learned positional embeddings, 6 pre-norm blocks, `dim=192`, 3 heads, MLP ratio 4; 2.71M parameters.
Classification uses the `[CLS]` token. With `distill=True` a `[DST]` token and a second head are added;
in training the model returns both heads' outputs, and in evaluation their average, as in the DeiT paper.

## Distillation losses

- **CE:** plain cross-entropy on the labels.
- **Soft (Hinton):** `α·T²·KL(teacher ‖ student) + (1−α)·CE`, with T=4 and α=0.9.
- **Hard (DeiT):** `½·CE(cls, y) + ½·CE(dst, argmax teacher)`.
- **Soft with distillation token:** `½·CE(cls, y) + ½·T²·KL(teacher ‖ dst)`, with T=4.

## Training setup

- **Data:** CIFAR-100; augmentation is RandomCrop(32, padding 4) and horizontal flip only (no Mixup/CutMix).
- **ResNets:** SGD (Nesterov, momentum 0.9), lr 0.1, weight decay 5e-4, cosine schedule; teacher 40 epochs, students 60 epochs.
- **ViT:** AdamW, lr 1e-3, weight decay 0.05 (not applied to biases, LayerNorm, tokens and positions), 5 warm-up epochs followed by a cosine schedule, 60 epochs.
- Batch size 128, seed 0, **a single run per configuration**. Training was done on a Kaggle GPU
  (about 4.4 h for the main experiment and about 46 min for the additional one).

## Results

Test accuracy (%) on CIFAR-100 after the last epoch:

| Model | Parameters | CE | Soft KD | Hard KD + dst token | Soft KD + dst token |
|---|---|---|---|---|---|
| Teacher ResNet-18 (w64) | 11.2M | 75.97 | | | |
| ResNet w32 | 2.8M | 74.37 | 76.46 | | |
| ResNet w16 | 0.7M | 69.57 | 72.83 | | |
| ViT | 2.7M | 55.61 | **63.81** | 56.64 | 62.43 |

For the last ViT variant the `[CLS]` head alone reaches 61.30, the `[DST]` head alone 63.37 and their
average 62.43 (`results/results_soft_dst.json`).

## Findings

1. **A ViT trained from scratch on 50k images is far behind a ResNet of the same size** (55.6 vs 74.4).
   It lacks the inductive bias of locality, and the data is small. The accuracy curve of the plain ViT
   plateaus after roughly 30 epochs.
2. **Soft distillation helps the weaker model the most:** +3.3 (ResNet w16), +2.1 (w32), +8.2 (ViT).
   The teacher is convolutional, so its soft outputs also transfer part of what the ViT cannot learn
   by itself. Distillation also speeds up convergence.
3. **Hard distillation with a distillation token yields little (+1.0)**, and its curve follows plain CE
   closely. A possible explanation, not verified: with weak augmentation the teacher is almost always
   correct on training images, so `argmax(teacher) ≈ y` and the distillation head receives almost no
   new information. DeiT uses strong augmentation, where the hard target carries more information.
4. **The distillation token itself is not what matters.** Soft distillation with a token (62.4) is not
   better than soft distillation without one (63.8), and the `[DST]` head alone (63.4) behaves almost
   like a plain soft-distilled ViT. The `[CLS]` head improved from 55.6 to 61.3 only because it shares
   the backbone with the `[DST]` head. Averaging the heads is worse than the `[DST]` head alone.
5. Distillation is more expensive: the soft runs take almost twice as long, since every step also
   requires a forward pass through the teacher.

## Limitations

- Each cell is a single seed. Differences of 1–2 points are within noise (0.1–0.6 points were observed
  between repeated ResNet runs earlier, and a ViT is likely less stable).
- The teacher was trained for 40 epochs and the students for 60, so the observation that ResNet w32
  with soft distillation exceeds the teacher (76.46 vs 75.97) is not a clean comparison.
- Hyperparameters were not tuned per variant. Plain soft distillation uses α=0.9 (90% of the signal
  from the teacher), while soft distillation with a token uses a 0.5/0.5 split, so those two are not
  strictly comparable.
- The explanation of finding 3 has not been tested directly. A suggested test is to repeat all three
  ViT variants with Mixup/CutMix.
- Preliminary CIFAR-10 runs (code no longer in this repository): teacher 93.6, student without
  distillation 82.9, with distillation 84.8.

## Usage

Training is intended for a GPU (Kaggle); locally, only shapes and losses can be checked reasonably.

```bash
uv sync
uv run python build_kaggle.py     # generates kaggle_c100.py and kaggle_soft_dst.py
```

On Kaggle: create a dataset from the `cifar-100-python` folder (to avoid downloading the data),
paste `kaggle_c100.py` into a single cell, enable the GPU and run *Save & Run All*. For the additional
experiment, add a dataset containing `teacher100.pt` (an output of the first run) and paste
`kaggle_soft_dst.py`. Trained weights are not stored in the repository (up to about 45 MB per model);
the numbers and curves are in `results/`.

## Next

DETR → Deformable DETR → camera geometry and BEV (LSS) → BEVFormer.
