# TopoSegNet

TopoSegNet is a controlled ablation study on [CloudSEN12](https://huggingface.co/datasets/csaybar/CloudSEN12-high). It segments Sentinel-2 L1C patches into four classes (`clear`, `thick cloud`, `thin cloud`, `cloud shadow`) and asks whether elevation helps. Every run uses a U-Net with an ImageNet-pretrained ResNet-34 encoder.

---

## TL;DR

**How elevation is delivered to the network matters, and the usual way of delivering it causes harm.** In a U-Net with a ResNet-34 encoder on CloudSEN12:

1. **Concatenating elevation as an extra encoder input channel lowers segmentation quality compared with using no terrain, and the loss grows with altitude.** It is concentrated above 3000 m.
2. **Injecting the same elevation through a separate branch at depth** (`gap`, `TCM`) **avoids that loss, but it does not beat the no-terrain baseline in any stratum.**
3. **Cross-attention adds nothing over its uniform-attention limit.** The terrain branch with attention deleted (`gap`) matches the cross-attention model (`TCM`), with 1.15% fewer parameters.

This is a cautionary result about *how* auxiliary inputs enter a network. It is **not** a claim that terrain improves segmentation, and **not** a new architecture.

---

## Key results

All numbers are test-set mIoU (975 patches), given as **mean [min, max] over seeds**. We report two checkpoint protocols: the best-validation checkpoint and the final (epoch-100) checkpoint. Statistics are **exact two-sided permutation tests over seed relabelings**.

### The headline: the altitude trend

The test statistic is the OLS slope of the per-band mIoU difference against the altitude-band index. Bands are 0–500, 500–1000, 1000–2000, 2000–3000 and >3000 m.

| comparison | best-val trend | p | final-epoch trend | p |
|---|---|---|---|---|
| `concat` − no terrain (`1a`) | −0.0069 / band | **0.029** | −0.0057 / band | 0.067 |
| `gap` − `concat` | +0.0053 / band | **0.029** | +0.0036 / band | 0.057 |

Per-band difference, `concat` − `1a` (best-val): −0.0003 · −0.0008 · −0.0042 · −0.0130 · **−0.0286**. Close to zero over the bottom 75% of pixels, and largest above 3000 m.

> ⚠️ **Statistical floor, stated up front.** With 4 vs 4 seeds there are only 70 relabelings, so the smallest achievable two-sided p is **0.029**. Both altitude trends reach that floor. After a Holm correction over the three pre-planned claims (aggregate, slope trend, altitude trend), the smallest achievable adjusted p is **0.086**. *Nothing passes Holm at α = 0.05.* That is a property of n = 4, not evidence that there is no effect.

### Above 3000 m (5.6% of test pixels), difference vs no terrain

| delivery route | best-val | final-epoch |
|---|---|---|
| `concat` (per-pixel elevation, input channel) | **−0.0286**, disjoint seed ranges, p = 0.029 | **−0.0218**, disjoint seed ranges |
| `elev_pooled` (patch-mean elevation, input channel) | **−0.0463**, disjoint seed ranges | **−0.0531**, disjoint seed ranges |
| `film_elev` (patch-mean elevation, bottleneck FiLM) | **−0.0163**, disjoint seed ranges | **−0.0151**, disjoint seed ranges |
| `TCM` (cross-attention branch) | −0.0009, ranges overlap | −0.0029, ranges overlap |
| `gap` (attention-free branch) | −0.0079, ranges overlap | −0.0088, ranges overlap |

Under both protocols, the input-channel routes (and also bottleneck FiLM) fall below the no-terrain baseline at high altitude. The two branch routes do not.

### Aggregate test mIoU

| config | n | best-val | final-epoch |
|---|---|---|---|
| `1a` — no terrain (13 bands) | 3 (final: 2) | 0.7005 [0.6999, 0.7012] | 0.6949 [0.6935, 0.6964] |
| `concat` | 4 | 0.6978 [0.6958, 0.6998] | 0.6933 [0.6880, 0.6963] |
| `elev_pooled` | 3 | 0.6939 [0.6888, 0.7024] | 0.6847 [0.6703, 0.7001] |
| `film_elev` | 4 | 0.6994 [0.6980, 0.7009] | 0.6970 [0.6953, 0.6992] |
| `TCM` | 3 | 0.7009 [0.6989, 0.7032] | 0.6959 [0.6941, 0.6985] |
| `gap` | 4 | 0.7008 [0.6974, 0.7036] | 0.6978 [0.6949, 0.6993] |

| comparison | best-val Δ, p | final-epoch Δ, p |
|---|---|---|
| `gap` − `concat` | +0.0031, p = 0.086 | +0.0045, p = 0.086 |
| `gap` − `1a` | +0.0003, p = 0.80 | +0.0029, p = 0.27 |
| `gap` − `TCM` | −0.0000, p = 1.00 | +0.0019, p = 0.29 |

Aggregate differences are small and none reaches p < 0.05. The slope trend has a consistent direction under both protocols but is not significant (`gap` − `concat` p = 0.143 / 0.200). The `1a` final-epoch band has only n = 2 because the original seedless baseline run did not save a final checkpoint.

---

## The delivery routes

Every terrain variant receives the **same elevation data**: the dataset's DEM plane, standardised with train-split statistics. The variants differ only in *where* and *in what form* the elevation enters the network. Hyperparameters are identical across all runs.

| `arch` | `terrain_mode` | what reaches the network | params |
|---|---|---|---|
| `unet` | `none` | 13 Sentinel-2 bands (baseline `1a`) | 24,487,218 |
| `unet` | `elev` | 13 bands + per-pixel elevation as a **14th input channel** (`concat`) | 24,490,354 |
| `unet` | `elev_pooled` | 13 bands + the **patch-mean** elevation, broadcast as a 14th input channel | 24,490,354 |
| `film_elev` | `elev_pooled` | 13-band encoder; patch-mean elevation → γ/β **FiLM on the 512-ch bottleneck** | 24,553,906 |
| `tcm` | `elev` | 13-band encoder; per-pixel elevation → terrain branch → **cross-attention** injected at encoder levels 4 and 5 | 25,163,178 |
| `gap` | `elev` | same as `tcm` with attention **deleted**: a global-average-pooled terrain embedding injected at levels 4 and 5 | 24,874,922 |

`gap` is the analytic uniform-attention limit of `TCM`. The value projection is a 1×1 conv, so it commutes with the spatial mean. A preflight check confirms the equivalence to 2.4e-7. All FiLM and attention outputs are zero-initialised, so every conditioned model starts as an exact copy of the 13-band baseline.

### Fixed training protocol

Adam · lr 1e-4 · ReduceLROnPlateau (patience 7) · batch 4 · 100 epochs · no augmentation · weighted cross-entropy (clear 1.00 / thick 2.04 / thin 5.65 / shadow 6.17).

Each ablation stage changes **only** `terrain_mode` and/or `arch`.

---

## What this repository does *not* claim

- ❌ "Terrain improves cloud segmentation." No route beats the no-terrain baseline in any stratum.
- ❌ "Cross-attention helps." It is indistinguishable from its uniform-attention limit, and the attention maps are near-uniform (normalised entropy 0.95–0.99).
- ❌ "Pooling is the mechanism." The same pooled scalar behaves differently depending on whether it is concatenated or injected.
- ❌ Any statement beyond **one architecture (U-Net + ResNet-34) and one dataset**. The effect is expected to depend on the architecture, and in particular on the skip connections that bypass any conditioning at depth (cf. [arXiv:1907.12330](https://arxiv.org/abs/1907.12330)).

---

## Ongoing follow-up: spectral ablation (pre-registered)

**Hypothesis (formed *after* the 13-band null, and reported as a follow-up):** terrain only adds information when the spectrum cannot separate the classes by itself. Sentinel-2's cirrus (B10) and SWIR (B11/B12) bands already separate snow from cloud. Earlier work that reported terrain gains (GeoInfoNet, Gaofen-1 WFV) used sensors with only blue/green/red/NIR bands.

**Test:** repeat the comparison on the four Gaofen-1-equivalent bands only (`rgbn` = B2, B3, B4, B8), at the same 10 m / 512×512 resolution.

| folder | role |
|---|---|
| `unet_rgbn_augFalse[_seed1]` | `none_4`: 4 bands, no terrain |
| `gap_rgbn_elev_augFalse[_seed1]` | `gap_4`: 4 bands + elevation via the gap branch (**primary**) |
| `unet_rgbn_elev_augFalse[_seed1]` | `concat_4`: 4 bands + elevation channel (exploratory) |

**Screening rule, fixed on 2026-10-10 before any 4-band run.** The primary comparison is `gap_4` − `none_4`, overall and at >3000 m, under both checkpoint protocols.

- **GO** (add seeds to reach n = 5): both `gap_4` seeds are above both `none_4` seeds, **and** the mean Δ at >3000 m is ≥ +0.010 (best-val), **and** that Δ is positive at final-epoch as well.
- **STOP**: mean Δ ≤ +0.002, both overall and at >3000 m, with overlapping ranges, under both protocols. In that case we report that the null persists even without cirrus or SWIR bands.
- **GREY**: anything else. Add one seed per config and re-apply the rule at n = 3.

An n = 2 screen is a go/no-go decision on effect size and consistency, never a significance claim. The verdict is computed mechanically by `analysis/spectral_screen.py`.

---

## Data

- **Dataset:** CloudSEN12, high-quality subset (Hugging Face [`csaybar/CloudSEN12-high`](https://huggingface.co/datasets/csaybar/CloudSEN12-high)). The provided split is **8,490 train / 535 val / 975 test**.
- **Patch size:** native patches are 509×509 (5.09 km), zero-padded by the providers to 512×512. We keep the 3-pixel border (≈1.2% of pixels) and it is included in all metrics.
- **Elevation:** the dataset's `EXTRA_elevation` layer. It comes from MERIT Hydro (~90 m), nearest-neighbour resampled to 10 m, which produces a "staircase" pattern. Slope and aspect derived from it are de-staircased first (see `src/regenerate_slope_aspect.py`).
- **Format:** raw `np.memmap` files of shape `(N, 512, 512)` per split.

```
data/high_509/{Train,Val,Test}/
├── L1C_B1.dat … L1C_B12.dat, L1C_B8A.dat   # 13 Sentinel-2 L1C bands (uint16)
├── EXTRA_elevation.dat                     # elevation in metres (int16)
├── DERIVED_slope_v2.dat                    # clean slope, degrees (float32)
├── DERIVED_{sin,cos}_aspect_v2.dat         # clean aspect (float32)
├── LABEL_manual_hq.dat                     # 0 clear · 1 thick · 2 thin · 3 shadow (int8)
└── metadata.csv
```

The Train/Val/Test regions do not overlap. The nearest cross-split patch is at least 38 km away; the median distance is about 200 km.

---

## Setup

Tested on Windows 11 with an RTX 4050 (6 GB), Python 3.10, PyTorch 2.5.1 + CUDA 12.1.

```bash
conda create -n toposegnet python=3.10
conda activate toposegnet
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install segmentation_models_pytorch==0.5.0 albumentations==2.0.8 rasterio==1.4.4 scipy==1.15.3 einops pandas matplotlib
```

`requirements.txt` is a full `pip freeze` of the development environment. Some of its lines point to local conda build paths, so use it as a version reference rather than installing it directly.

**Known constraints:**
- **`num_workers=0` in every DataLoader.** Pickling memmaps breaks under Windows multiprocessing spawn.
- **Paths are hardcoded and absolute** (`DATA_ROOT` in `src/dataset.py`, `save_dir` in `train.py`, and in the analysis scripts). Update them for your machine.
- `src/evaluate_notebook.py` needs `PYTHONIOENCODING=utf-8` when stdout is piped.

---

## Usage

### Train one configuration

Edit the `CONFIG` dict at the top of `train.py` (`terrain_mode`, `arch`, `seed`), or override it with JSON:

```bash
TOPOSEG_CONFIG_OVERRIDE='{"terrain_mode": "elev", "arch": "gap", "seed": 1}' python train.py
```

Each run writes to `experiments/{arch}_{terrain_mode}_aug{augment}[_seed{N}]/`:
- `config.json` and `history.json`, which logs loss, mIoU, per-class IoU and learning rate for every epoch
- `best_model.pth`
- `last_checkpoint.pth`, a full model + optimizer + scheduler state for resuming

Set `"resume": true` to continue an interrupted run. Starting a fresh run in a folder that already has a checkpoint archives the old folder instead of overwriting it.

### Evaluate on the test set

```python
import sys; sys.path.append("src")
from evaluate_notebook import run_test_eval
run_test_eval("elev", "experiments/gap_elev_augFalse/best_model.pth", arch="gap")
```

### Run a queue of experiments

```bash
python run_queue.py
```

`run_queue.py` runs the `STEPS` list in order: training, then evaluation of both the best-val and final-epoch checkpoints, then the analysis scripts. You can kill it and re-launch at any time. It skips finished work and resumes interrupted runs.

### Analysis (no GPU needed after stratification)

| script | output |
|---|---|
| `analysis/stratified_model_comparison.py` | Per-model confusion matrices by surface and altitude band (set `TOPOSEG_STRAT_CKPT=final` for the final-epoch protocol) |
| `analysis/joint_slope_elev_stratification.py` | Confusion matrices by slope bin, and by slope × surface |
| `analysis/snow_free_report.py` | Overall / altitude / slope tables using strata derived only from the DEM |
| `analysis/trend_permutation_test.py` | The pre-planned exact permutation tests with Holm correction → `experiments/stratified_comparison/trend_tests.json` |
| `analysis/tcm_attention_analysis.py`, `tcm_attention_summary.py` | Attention entropy, locality and elevation-organisation statistics for `TCM` |
| `analysis/preflight_*.py` | CPU checks run before each new configuration: input contracts, zero-init identity, live conditioning paths, parameter accounting |

---

## Repository layout

```
├── train.py                 # training entry point (CONFIG at top)
├── run_queue.py             # kill-safe experiment queue
├── src/
│   ├── dataset.py           # CloudSEN12Dataset; terrain_mode → input channels
│   ├── model.py             # baseline U-Net, FiLM, TCM, gap and gap_latlon models
│   ├── tcm.py               # cross-attention terrain conditioning module
│   ├── evaluate_notebook.py # test-set evaluator
│   └── regenerate_slope_aspect.py  # de-staircased slope/aspect (v2)
├── analysis/                # stratification, statistics, attention analysis, preflights
└── experiments/             # per-run outputs (not tracked)
```

---

## Related work this builds on

- **FiLM**: Perez et al., [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
- **Conditioning segmentation networks with non-imaging data**: [arXiv:1907.12330](https://arxiv.org/abs/1907.12330). Finds that conditioning a standard U-Net is hard, and points to the skip-connection bypass.
- **Concat vs FiLM vs cross-attention for segmentation**: [arXiv:2304.14371](https://arxiv.org/abs/2304.14371). Finds cross-attention best; it does not include a uniform-attention ablation.
- **PAPA** (replacing attention with constant averaged attention): [arXiv:2211.03495](https://arxiv.org/abs/2211.03495)
- **CloudSEN12**: Aybar et al., *Scientific Data* 2022, [doi:10.1038/s41597-022-01878-2](https://doi.org/10.1038/s41597-022-01878-2)
- **MERIT Hydro**: Yamazaki et al., *Water Resources Research* 2019
