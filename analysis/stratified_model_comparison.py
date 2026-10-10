"""
Stratified model comparison — does conditioning help WHERE TERRAIN SHOULD MATTER?
================================================================================


Why this exists
---------------
Aggregate test mIoU has failed to separate any variant in the terrain chain:
everything sits in 0.685-0.705, and the measured checkpoint-selection noise
(~0.003 test mIoU from a 2.3e-06 val difference, see project notes) is the same
size as the effects we are chasing. Aggregate mIoU is also dominated by the
`clear` and `thick` classes over ordinary terrain — precisely the pixels where
terrain conditioning should make NO difference. A real effect confined to snow
or shadow is invisible there.

The Stage 1a error analysis identified the actual headroom:
  * snow = 7.3% of test pixels, error 16.2% vs 10.7% elsewhere
  * over snow: thick IoU 0.61 (vs 0.86), thin 0.33 (vs 0.49)
  * SHADOW RECALL over snow 0.61 (vs 0.79) — shadow-on-snow is called `clear`
    ~24% of the time. This is the physically terrain-driven failure.

So the question this script answers is NOT "which model has the best mIoU" but:
  >>> Does terrain/geo conditioning reduce the SNOW-SPECIFIC failure, and in
      particular does it recover shadow-on-snow?

A model that is flat on aggregate mIoU but clearly better on snow-shadow is a
far stronger result for the TCM argument than +0.004 overall — and it is the
story the error analysis predicts.

What it does
------------
For each checkpoint: runs test-set inference and accumulates 4x4 confusion
matrices binned by surface type (other / snow / water_like) and by elevation.
Then reports per-stratum IoU / recall / precision per class, plus deltas
against the chosen baseline model.

Surface definitions are IDENTICAL to stage1a_terrain_error_analysis.py
(NDSI > 0.4 split by NIR) so numbers are directly comparable to that analysis.
Masks come from the 13 spectral bands, which are byte-identical across every
terrain_mode, so all models are stratified on exactly the same pixels.

Run:  python analysis/stratified_model_comparison.py
      (conda env `toposegnet`; ~2 min per model on GPU at batch=4)
"""

import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import CloudSEN12Dataset
from dataset import S2_SCALE   # strata read raw bands with the dataset's own scaling
from model import build_film_unet

EXP = r"C:\Users\prash\toposegnet\experiments"
OUT_DIR = os.path.join(EXP, "stratified_comparison")
# Which checkpoint to stratify. Default best_model.pth (the chain's protocol). With
# TOPOSEG_STRAT_CKPT=final, last_checkpoint.pth (epoch 100) is stratified instead and
# results go to separate *_final.json files, so the final-epoch robustness check
# (Statistics item 2) never reads or overwrites the best-val results.
CKPT_FILE  = "last_checkpoint.pth" if os.environ.get("TOPOSEG_STRAT_CKPT") == "final" else "best_model.pth"
OUT_SUFFIX = "_final" if CKPT_FILE == "last_checkpoint.pth" else ""


def training_unfinished(spec, epochs=100):
    """True for a resumable run that is still mid-training.

    A run being trained has a best_model.pth from its best epoch SO FAR, so
    without this guard a stratification pass would score a half-trained model and
    the confusion cache would keep reusing it after training finishes. Only runs
    WITH a last_checkpoint.pth are judged: pre-resume-support runs (seedless 1a,
    3-v2, 2a) have none, and their histories legitimately end at ep 91-96.

    """
    d = os.path.join(EXP, spec["dir"])
    hist = os.path.join(d, "history.json")
    if not (os.path.exists(os.path.join(d, "last_checkpoint.pth")) and os.path.exists(hist)):
        return False
    with open(hist) as f:
        return max((x["epoch"] for x in json.load(f)), default=0) < epochs

# ── models to compare ────────────────────────────────────────────────────────
# arch: "unet" = all dataset channels into the encoder (naive concat)
#       "film" = 13 bands into encoder, lat/lon peeled off into the FiLM MLP
# `baseline` marks the reference the deltas are computed against.
MODELS = [
    dict(name="1a baseline (13ch)",      dir="unet_none_augFalse",
         terrain_mode="none",        arch="unet", baseline=True),
    dict(name="3-v2 +slope (concat)",    dir="unet_optionA_v2_augFalse",
         terrain_mode="optionA_v2",  arch="unet"),
    dict(name="5b(1) latlon CONCAT",     dir="unet_latlon_only_augFalse",
         terrain_mode="latlon_only", arch="unet"),
    dict(name="5b(2) latlon FiLM",       dir="film_latlon_only_augFalse",
         terrain_mode="latlon_only", arch="film"),
    # Stage 6 and its same-feature comparator. 2a and 6 consume the byte-identical
    # 14ch `elev` tensor; only the injection differs (concat vs cross-attention),
    # which is the MSCloudCAM-style comparison.
    dict(name="2a elev CONCAT",          dir="unet_elev_augFalse",
         terrain_mode="elev",        arch="unet"),
    dict(name="6 elev CROSS-ATTN (TCM)", dir="tcm_elev_augFalse",
         terrain_mode="elev",        arch="tcm"),
    # Seed replicate — the snow effect at seed 0 is the chain's only positive
    # result, so it has to reproduce before anything is claimed.
    dict(name="6 TCM seed1",             dir="tcm_elev_augFalse_seed1",
         terrain_mode="elev",        arch="tcm"),
    # Campaign completion: n=3 per side so concat-vs-attention finally gets error
    # bars. The seedless 2a above is NOT a controlled replicate (it predates
    # seeding), so concat's three controlled seeds are 1/2/3 and TCM's are 0/1/2.
    dict(name="6 TCM seed2",             dir="tcm_elev_augFalse_seed2",
         terrain_mode="elev",        arch="tcm"),
    dict(name="2a concat seed1",         dir="unet_elev_augFalse_seed1",
         terrain_mode="elev",        arch="unet"),
    dict(name="2a concat seed2",         dir="unet_elev_augFalse_seed2",
         terrain_mode="elev",        arch="unet"),
    dict(name="2a concat seed3",         dir="unet_elev_augFalse_seed3",
         terrain_mode="elev",        arch="unet"),
    # gap_terrain ablation: TCM with attention deleted (analytic uniform-attention
    # limit, 1.15% FEWER params). Same 14ch elev input as concat and TCM, so this
    # is a three-way A/B on injection method alone.
    dict(name="gap seed0",               dir="gap_elev_augFalse",
         terrain_mode="elev",        arch="gap"),
    dict(name="gap seed1",               dir="gap_elev_augFalse_seed1",
         terrain_mode="elev",        arch="gap"),
    dict(name="gap seed2",               dir="gap_elev_augFalse_seed2",
         terrain_mode="elev",        arch="gap"),
    # Pooled-elevation control: gap differs from concat in TWO ways at once
    # (pooled-not-per-pixel AND separate-branch-not-input-channel). This holds the
    # delivery route fixed — still a concatenated 14th channel, still a plain
    # U-Net — and varies ONLY pooling, so the two are separable.
    dict(name="elev_pooled seed0",       dir="unet_elev_pooled_augFalse",
         terrain_mode="elev_pooled", arch="unet"),
    dict(name="elev_pooled seed1",       dir="unet_elev_pooled_augFalse_seed1",
         terrain_mode="elev_pooled", arch="unet"),
    dict(name="elev_pooled seed2",       dir="unet_elev_pooled_augFalse_seed2",
         terrain_mode="elev_pooled", arch="unet"),
    # Pooled-elevation FiLM control: the SAME patch-mean scalar as elev_pooled
    # above, but peeled at pixel (0,0) and injected as gamma/beta at the
    # bottleneck instead of entering the encoder as a 14th channel. Separates
    # "pooling does not help" from "a concatenated channel cannot exploit
    # pooling", which the elev_pooled null could not.
    dict(name="film_elev seed0",         dir="film_elev_elev_pooled_augFalse",
         terrain_mode="elev_pooled", arch="film_elev"),
    dict(name="film_elev seed1",         dir="film_elev_elev_pooled_augFalse_seed1",
         terrain_mode="elev_pooled", arch="film_elev"),
    dict(name="film_elev seed2",         dir="film_elev_elev_pooled_augFalse_seed2",
         terrain_mode="elev_pooled", arch="film_elev"),
    # Seeded 1a replicates (Statistics item 0): the seedless original above stays
    # the `baseline` for the delta printout, but "does terrain help at all?" needs a
    # no-terrain band, not a single run.
    dict(name="1a seed1",                dir="unet_none_augFalse_seed1",
         terrain_mode="none",        arch="unet"),
    dict(name="1a seed2",                dir="unet_none_augFalse_seed2",
         terrain_mode="none",        arch="unet"),
    # 4th replicate per primary config (Statistics item 3: n=4 lowers the exact
    # permutation p floor from 0.10 to 0.029). Concat already uses seeds 1/2/3, so
    # its 4th is seed 4. Skipped automatically until each checkpoint exists.
    dict(name="2a concat seed4",         dir="unet_elev_augFalse_seed4",
         terrain_mode="elev",        arch="unet"),
    dict(name="film_elev seed3",         dir="film_elev_elev_pooled_augFalse_seed3",
         terrain_mode="elev_pooled", arch="film_elev"),
    dict(name="gap seed3",               dir="gap_elev_augFalse_seed3",
         terrain_mode="elev",        arch="gap"),
    # Spectral-ablation follow-up, Phase 1 n=2 screen (project notes ACTIVE PLAN):
    # the same three routes at the 4 Gaofen-1-equivalent bands. none_4 = no
    # terrain, gap_4 = elevation via the gap branch (PRIMARY), concat_4 =
    # elevation as a 5th input channel (exploratory).
    dict(name="none_4 seed0",            dir="unet_rgbn_augFalse",
         terrain_mode="rgbn",        arch="unet"),
    dict(name="none_4 seed1",            dir="unet_rgbn_augFalse_seed1",
         terrain_mode="rgbn",        arch="unet"),
    dict(name="gap_4 seed0",             dir="gap_rgbn_elev_augFalse",
         terrain_mode="rgbn_elev",   arch="gap"),
    dict(name="gap_4 seed1",             dir="gap_rgbn_elev_augFalse_seed1",
         terrain_mode="rgbn_elev",   arch="gap"),
    dict(name="concat_4 seed0",          dir="unet_rgbn_elev_augFalse",
         terrain_mode="rgbn_elev",   arch="unet"),
    dict(name="concat_4 seed1",          dir="unet_rgbn_elev_augFalse_seed1",
         terrain_mode="rgbn_elev",   arch="unet"),
]

BATCH_SIZE  = 4
NUM_CLASSES = 4
CLASS_NAMES = ["clear", "thick", "thin", "shadow"]
N_TEST      = 975

# must match train.py / evaluate_notebook.py
IN_CHANNELS = {
    "none": 13, "elev": 14, "optionA": 15, "optionB": 17, "optionC": 19,
    "optionA_v2": 15, "optionB_v2": 17, "optionC_v2": 19, "latlon_only": 15,
    "elev_pooled": 14,   # pooled-elevation control
    "rgbn": 4, "rgbn_elev": 5,   # spectral-ablation follow-up, 4 GF-1-equivalent bands
}

# ── surface proxy — IDENTICAL to stage1a_terrain_error_analysis.py ───────────
# NDSI = (B3 - B11)/(B3 + B11) > 0.4, split by NIR because water is ALSO
# high-NDSI. Snow is NIR-bright, water is NIR-dark. Band idx: B3=2, B8=7, B11=11.
NDSI_B3_IDX, NDSI_B8_IDX, NDSI_B11_IDX = 2, 7, 11
NDSI_THRESH, NIR_THRESH = 0.4, 0.11
SURF_LABELS = ["other", "snow", "water_like"]

ELEV_EDGES  = np.array([-np.inf, 0, 500, 1000, 2000, 3000, np.inf])
ELEV_LABELS = ["<0", "0-500", "500-1000", "1000-2000", "2000-3000", ">3000"]

DATA_ROOT = r"C:\Users\prash\toposegnet\data\high_509"


# The snow / water_like strata must come from the FULL 13-band data, never from
# the model's input tensor: the spectral-ablation runs (rgbn, rgbn_elev) feed only
# B2/B3/B4/B8, so channel 11 (B11) does not exist in their input. Reading B3/B8/B11
# straight from the Test memmaps gives every model exactly the same strata. Same
# float32 arithmetic as CloudSEN12Dataset (uint16 -> float32 / S2_SCALE), so the
# stratification of every existing 13-band model is unchanged — asserted in
# analysis/preflight_rgbn.py.
_SPEC_MM = {}


def spectral_bands(i0, n):
    """(b3, b8, b11) for test patches i0 .. i0+n-1, each (n, 512, 512) float32."""
    if not _SPEC_MM:
        for key, fname in [("b3", "L1C_B3.dat"), ("b8", "L1C_B8.dat"), ("b11", "L1C_B11.dat")]:
            _SPEC_MM[key] = np.memmap(f"{DATA_ROOT}\\Test\\{fname}", dtype="uint16",
                                      mode="r", shape=(N_TEST, 512, 512))
    return tuple(_SPEC_MM[k][i0:i0 + n].astype(np.float32) / S2_SCALE for k in ("b3", "b8", "b11"))


def load_model(spec, device):
    """Build the right architecture and load weights (handles BOTH ckpt formats)."""
    ds_ch = IN_CHANNELS[spec["terrain_mode"]]
    if spec["arch"] == "tcm":
        # `elev` is 14ch: encoder takes the 13 bands, elevation goes to the
        # cross-attention terrain branch.
        from model import build_tcm_unet
        model = build_tcm_unet(in_channels=ds_ch - 1, num_classes=NUM_CLASSES,
                               pretrained=False)
    elif spec["arch"] == "gap":
        # gap_terrain: same 13ch encoder split as tcm; elevation enters via a
        # global-average-pooled terrain embedding.
        from model import build_gap_terrain_unet
        model = build_gap_terrain_unet(in_channels=ds_ch - 1, num_classes=NUM_CLASSES,
                                       pretrained=False)
    elif spec["arch"] == "film":
        # encoder sees bands only; lat/lon enter via the FiLM MLP
        model = build_film_unet(in_channels=ds_ch - 2, num_classes=NUM_CLASSES,
                                pretrained=False)
    elif spec["arch"] == "film_elev":
        # Pooled-elevation FiLM: `elev_pooled` is 14ch, the encoder takes the 13
        # bands, and the ONE constant elevation plane is peeled at pixel (0,0)
        # and injected as gamma/beta at the bottleneck — hence -1, not -2.
        # Without this branch the spec would silently fall through to a plain
        # 14ch U-Net and the state_dict load would fail.
        from model import build_film_elev_unet
        model = build_film_elev_unet(in_channels=ds_ch - 1, num_classes=NUM_CLASSES,
                                     pretrained=False)
    else:
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=ds_ch, classes=NUM_CLASSES)
    ckpt = torch.load(os.path.join(EXP, spec["dir"], CKPT_FILE),   # best-val or final epoch, see CKPT_FILE
                      map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state, epoch = ckpt["model_state_dict"], ckpt.get("epoch")
    else:
        state, epoch = ckpt, None            # old stage 1-3 bare state_dict
    model.load_state_dict(state)
    return model.to(device).eval(), epoch


def bin_index(values, edges):
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def metrics_from_conf(conf):
    """conf[true, pred] -> per-class IoU / recall / precision."""
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp          # true c, predicted something else
        fp = conf[:, c].sum() - tp          # predicted c, actually something else
        union = tp + fp + fn
        out[name] = dict(
            iou       = float(tp / union)        if union else float("nan"),
            recall    = float(tp / (tp + fn))    if (tp + fn) else float("nan"),
            precision = float(tp / (tp + fp))    if (tp + fp) else float("nan"),
            support   = int(conf[c, :].sum()),
        )
    ious = [out[n]["iou"] for n in CLASS_NAMES if not np.isnan(out[n]["iou"])]
    out["miou"] = float(np.mean(ious)) if ious else float("nan")
    return out


@torch.no_grad()
def evaluate(spec, device, elev_mm):
    """Run one model over the test set, accumulating stratified confusions."""
    model, epoch = load_model(spec, device)
    ds = CloudSEN12Dataset(split="Test", terrain_mode=spec["terrain_mode"], augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)   # HARD: num_workers=0

    conf_all  = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_surf = np.zeros((3, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_elev = np.zeros((len(ELEV_LABELS), NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for b, (images, labels) in enumerate(tqdm(loader, desc=f"  {spec['name'][:28]:28s}",
                                              leave=False)):
        preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
        y_np  = labels.numpy()
        i0    = b * BATCH_SIZE                    # shuffle=False -> idx is positional

        # Strata from the 13-band memmaps, not the input tensor, so 4-band models
        # are stratified on exactly the same pixels as everything else.
        # (previously read images[:, NDSI_*_IDX])
        b3, b8, b11 = spectral_bands(i0, preds.shape[0])
        ndsi = (b3 - b11) / (b3 + b11 + 1e-8)
        hi   = ndsi > NDSI_THRESH
        surf = (hi & (b8 > NIR_THRESH)).astype(np.int64) \
             + 2 * (hi & (b8 <= NIR_THRESH)).astype(np.int64)

        for k in range(preds.shape[0]):
            idx = i0 + k
            yp  = (y_np[k].ravel() * NUM_CLASSES + preds[k].ravel()).astype(np.int64)
            conf_all  += np.bincount(yp, minlength=16).reshape(4, 4)
            conf_surf += np.bincount(surf[k].ravel() * 16 + yp,
                                     minlength=3 * 16).reshape(3, 4, 4)
            eb = bin_index(elev_mm[idx].astype(np.float32).ravel(), ELEV_EDGES)
            conf_elev += np.bincount(eb * 16 + yp,
                                     minlength=len(ELEV_LABELS) * 16
                                     ).reshape(len(ELEV_LABELS), 4, 4)

    total_px = N_TEST * 512 * 512
    assert conf_all.sum() == total_px, "lost pixels in overall confusion"
    assert conf_surf.sum() == total_px, "lost pixels in surface confusion"

    del model
    torch.cuda.empty_cache()
    return dict(epoch=epoch, conf_all=conf_all, conf_surf=conf_surf, conf_elev=conf_elev)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    elev_mm = np.memmap(f"{DATA_ROOT}\\Test\\EXTRA_elevation.dat",
                        dtype="int16", mode="r", shape=(N_TEST, 512, 512))

    # Reuse confusions already saved in stratified_comparison.json for models whose
    # name AND dir are unchanged, so adding runs costs only their own inference
    # instead of re-running all ~20 models (~2 h, and lost entirely if the machine
    # sleeps). Delete the JSON to force a full recompute.
    cache = {}
    prev_path = os.path.join(OUT_DIR, f"stratified_comparison{OUT_SUFFIX}.json")   # cache per checkpoint type
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            cache = json.load(f).get("models", {})

    results, specs = {}, []
    for spec in MODELS:
        ck = os.path.join(EXP, spec["dir"], CKPT_FILE)   # see CKPT_FILE
        if not os.path.exists(ck):
            print(f"SKIP {spec['name']} — no checkpoint at {ck}")
            continue
        if training_unfinished(spec):   # never score/cache a model mid-training
            print(f"SKIP {spec['name']} — training not finished")
            continue
        c = cache.get(spec["name"])
        if c and c.get("dir") == spec["dir"] and "conf_elev" in c:
            res = dict(epoch=c["checkpoint_epoch"],
                       conf_all=np.array(c["conf_all"], dtype=np.int64),
                       conf_surf=np.array(c["conf_surf"], dtype=np.int64),
                       conf_elev=np.array(c["conf_elev"], dtype=np.int64))
        else:
            res = evaluate(spec, device, elev_mm)
        results[spec["name"]] = res
        specs.append(spec)
        print(f"done {spec['name']:28s} (ckpt ep{res['epoch']})  "
              f"overall mIoU {metrics_from_conf(res['conf_all'])['miou']:.4f}")

    if not results:
        print("No checkpoints found — nothing to compare.")
        return

    base_name = next((s["name"] for s in specs if s.get("baseline")), specs[0]["name"])

    # ── report ───────────────────────────────────────────────────────────────
    def block(title, conf_getter, note=""):
        print("\n" + "=" * 100)
        print(title + ("   " + note if note else ""))
        print("=" * 100)
        hdr = f"{'model':30s} {'mIoU':>8s}" + "".join(f"{n:>10s}" for n in CLASS_NAMES) \
              + f"{'shadowRec':>11s}"
        print(hdr); print("-" * len(hdr))
        base_m = metrics_from_conf(conf_getter(results[base_name]))
        for s in specs:
            m = metrics_from_conf(conf_getter(results[s["name"]]))
            row = f"{s['name']:30s} {m['miou']:8.4f}"
            row += "".join(f"{m[n]['iou']:10.4f}" for n in CLASS_NAMES)
            row += f"{m['shadow']['recall']:11.4f}"
            print(row)
            if s["name"] != base_name:
                d = f"{'  ^ delta vs baseline':30s} {m['miou']-base_m['miou']:+8.4f}"
                d += "".join(f"{m[n]['iou']-base_m[n]['iou']:+10.4f}" for n in CLASS_NAMES)
                d += f"{m['shadow']['recall']-base_m['shadow']['recall']:+11.4f}"
                print(d)

    n_surf_px = results[base_name]["conf_surf"].sum(axis=(1, 2))
    tot = n_surf_px.sum()

    block("OVERALL (all pixels)", lambda r: r["conf_all"])
    for si, lab in enumerate(SURF_LABELS):
        block(f"SURFACE = {lab.upper()}", lambda r, i=si: r["conf_surf"][i],
              note=f"({n_surf_px[si]/tot*100:.1f}% of test pixels)")

    # ── the specific failure the error analysis identified ───────────────────
    print("\n" + "=" * 100)
    print("SHADOW-ON-SNOW  (the physically terrain-driven failure mode)")
    print("=" * 100)
    print(f"{'model':30s} {'shadowRec':>11s} {'->clear':>10s} {'->thick':>10s} {'->thin':>10s}")
    print("-" * 73)
    for s in specs:
        c = results[s["name"]]["conf_surf"][1]      # snow stratum
        true_shadow = c[3, :].sum()
        if true_shadow == 0:
            print(f"{s['name']:30s}  (no shadow pixels over snow)"); continue
        print(f"{s['name']:30s} {c[3,3]/true_shadow:11.4f} "
              f"{c[3,0]/true_shadow:10.4f} {c[3,1]/true_shadow:10.4f} {c[3,2]/true_shadow:10.4f}")
    print("\n  Stage 1a reference: shadow recall over snow 0.61 vs 0.79 elsewhere;")
    print("  shadow-on-snow mislabelled `clear` ~24% of the time.")
    print("  A conditioning win should show up HERE first, not in overall mIoU.")

    # ── save ─────────────────────────────────────────────────────────────────
    payload = {
        "class_names": CLASS_NAMES, "surf_labels": SURF_LABELS,
        "elev_labels": ELEV_LABELS, "baseline": base_name,
        "ndsi_thresh": NDSI_THRESH, "nir_thresh": NIR_THRESH,
        "models": {},
    }
    for s in specs:
        r = results[s["name"]]
        payload["models"][s["name"]] = {
            "dir": s["dir"], "arch": s["arch"], "terrain_mode": s["terrain_mode"],
            "checkpoint_epoch": r["epoch"],
            "overall": metrics_from_conf(r["conf_all"]),
            "by_surface": {lab: metrics_from_conf(r["conf_surf"][i])
                           for i, lab in enumerate(SURF_LABELS)},
            "by_elevation": {lab: metrics_from_conf(r["conf_elev"][i])
                             for i, lab in enumerate(ELEV_LABELS)},
            "conf_all": r["conf_all"].tolist(),
            "conf_surf": r["conf_surf"].tolist(),
            "conf_elev": r["conf_elev"].tolist(),
        }
    out = os.path.join(OUT_DIR, f"stratified_comparison{OUT_SUFFIX}.json")   # separate file for final-epoch
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
