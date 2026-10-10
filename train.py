import sys
sys.path.append(r"C:\Users\prash\toposegnet\src")

import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import json

from dataset import CloudSEN12Dataset
from model import build_film_unet   # Stage 5b(2) FiLM model
from model import build_multiscale_film_unet   # Stage 5b(3) multi-scale FiLM
from model import build_tcm_unet   # Stage 6 TCM cross-attention
from model import build_gap_terrain_unet   # gap_terrain ablation: TCM minus attention

# ── config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # "terrain_mode" selects the FEATURE; "arch" selects the INJECTION METHOD.
    # Holding one fixed while varying the other is what makes each comparison clean:
    #   elev        + arch="unet" -> Stage 2a, elevation via naive concat (test 0.7009)
    #   elev        + arch="tcm"  -> Stage 6,  elevation via cross-attention  <-- NOW
    #   latlon_only + arch="unet" -> Stage 5b(1) concat | + arch="film" -> 5b(2)
    "terrain_mode" : "elev_latlon",   # gap + lat/lon FiLM, exploratory single seed: 16ch = bands + per-pixel elev + lat/lon
                              # was "elev_pooled": POOLED-ELEVATION CONTROL (2026-08-13): 14ch, elevation
                              # plane = patch mean. With arch="unet" this holds the delivery
                              # route fixed (concatenated input channel) and varies ONLY
                              # pooled-vs-per-pixel, separating the two things gap changed at once.
    "arch"         : "gap_latlon",  # elevation -> gap branch, lat/lon -> bottleneck FiLM; tests whether location adds to depth-injected terrain
                              # was "film_elev". MULTI-SEED CAMPAIGN (2026-08-01). The Stage 6 snow result did
                              # NOT replicate: snow shadow recall was 0.6440 at seed 0 but 0.5802
                              # at seed 1, vs a 1a baseline of 0.6126 — seed spread (~0.064) is
                              # ~20x the checkpoint-noise floor and swamps the effect. Now running
                              # n=3 per config to put real error bars on the concat-vs-attention
                              # comparison. 2a concat: seeds 1,2,3. TCM: seeds 0,1 done + seed 2.
                              # The one effect that DID reproduce is thick-cloud IoU on snow
                              # (+0.0204 / +0.0230 across both TCM seeds) — that is what these
                              # runs are really characterising.
    "seed"         : 0,       # single-seed look; seed 0 -> unsuffixed new folder gap_latlon_elev_latlon_augFalse
                              # previously 2: pooled-elevation FiLM control, replicate 3/3
                              # gap_terrain ablation, replicate 1/3. Seeds 0,1,2 mirror the TCM
                              # side; the folder is gap_elev_augFalse (no suffix at seed 0), which
                              # collides with nothing.
                              # seeded from 5b(2) on: 5b(2) is the first run where we want a
                              # POSITIVE result, so it must be reproducible. Seeding fixes which
                              # draw we get, not the distribution — no bias vs seedless 1a-5.
    "augment"      : False,
    "epochs"       : 100,
    "batch_size"   : 4,
    "lr"           : 1e-4,
    "num_workers"  : 0,
    "save_dir"     : r"C:\Users\prash\toposegnet\experiments",
    "resume"       : False,  # back to default: gap_latlon seed0 resume (ep95-100, after the 2026-10-02 app-update kill) completed; resume=True bypasses the overwrite guard
}
# run_queue.py launches queued runs (e.g. 1a seeds back to back) by passing a JSON
# override here instead of editing CONFIG by hand; absent the env var, CONFIG above
# is used unchanged.
if os.environ.get("TOPOSEG_CONFIG_OVERRIDE"):
    CONFIG.update(json.loads(os.environ["TOPOSEG_CONFIG_OVERRIDE"]))
def atomic_torch_save(obj, path):
    """Write a checkpoint so a kill can never leave a half-written file.

    Four app updates killed runs mid-training (2026-10-02..08). A kill
    during a plain torch.save would truncate last_checkpoint.pth and make the
    run unresumable; writing to a temp file and os.replace()-ing it means the
    old checkpoint survives until the new one is complete.

    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ── class weights (computed from full 8490-patch train set) ──────────────────
CLASS_WEIGHTS = torch.tensor([1.0000, 2.0384, 5.6464, 6.1658], dtype=torch.float32)

# ── input channels per terrain mode ─────────────────────────────────────────
IN_CHANNELS = {
    "none"   : 13,
    "elev"   : 14,
    "optionA": 15,
    "optionB": 17,
    "optionC": 19,
    # v2 modes: same channel counts, clean slope/aspect data
    "optionA_v2": 15,
    "optionB_v2": 17,
    "optionC_v2": 19,
    "latlon_only": 15,  # Stage 5b(1): 13 bands + lat + lon, no other terrain
    "elev_pooled": 14,  # pooled-elevation control: same 14ch as elev, elevation plane = patch mean
    "elev_latlon": 16,  # 13 bands + per-pixel elev + lat + lon, for arch="gap_latlon"
    "rgbn": 4,          # spectral-ablation follow-up: B2,B3,B4,B8 only
    "rgbn_elev": 5,     # the same 4 bands + per-pixel elevation; gap peels the 5th plane generically
}

# ── metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(preds, labels, num_classes=4):
    """
    preds  : (N, H, W) long tensor — predicted class indices
    labels : (N, H, W) long tensor — ground truth class indices
    returns: dict with per-class IoU, mIoU, per-class F1, mF1
    """
    iou_per_class = []
    f1_per_class  = []

    for c in range(num_classes):
        pred_c  = (preds  == c)
        label_c = (labels == c)

        tp = (pred_c & label_c).sum().item()
        fp = (pred_c & ~label_c).sum().item()
        fn = (~pred_c & label_c).sum().item()

        iou = tp / (tp + fp + fn + 1e-8)
        f1  = 2*tp / (2*tp + fp + fn + 1e-8)

        iou_per_class.append(iou)
        f1_per_class.append(f1)

    return {
        "iou"  : iou_per_class,
        "miou" : np.mean(iou_per_class),
        "f1"   : f1_per_class,
        "mf1"  : np.mean(f1_per_class),
    }

# ── train one epoch ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)          # (B, C, 512, 512)
        labels = labels.to(device).long()   # (B, 512, 512)

        optimizer.zero_grad()
        logits = model(images)              # (B, 4, 512, 512)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ── validate one epoch ───────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)        # (B, 512, 512)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics    = compute_metrics(all_preds, all_labels)

    return total_loss / len(loader), metrics

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : {CONFIG}")

    # Seed weight init + shuffle order so the run is reproducible. Not bit-exact
    # on GPU (cudnn picks nondeterministic kernels) but fixes the draw that
    # matters.
    if CONFIG.get("seed") is not None:
        torch.manual_seed(CONFIG["seed"])
        np.random.seed(CONFIG["seed"])
        print(f"Seed   : {CONFIG['seed']}")

    # ── experiment folder ────────────────────────────────────────────────────
    # arch prefix keeps Stage 5b(2) in its own folder; default "unet" preserves
    # every existing 1a-5 folder name.
    exp_name = f"{CONFIG.get('arch', 'unet')}_{CONFIG['terrain_mode']}_aug{CONFIG['augment']}"
    # Multi-seed replicates get their own folder. Suffix ONLY for seed != 0, so
    # every existing seed-0 / seedless folder keeps its name.
    if CONFIG.get("seed"):
        exp_name += f"_seed{CONFIG['seed']}"
    save_dir = os.path.join(CONFIG["save_dir"], exp_name)

    # ── overwrite guard ──────────────────────
    # A fresh (resume=False) launch into a folder that already holds results
    # clobbers best_model.pth/history.json on epoch 1 (best_miou resets to 0.0).
    # This is exactly what destroyed the Stage 5 optionC_v2 training history.
    # Guard: if NOT resuming and a prior run exists, archive the whole folder to
    # a timestamped sibling before starting fresh — never silently overwrite.
    if (not CONFIG.get("resume")) and os.path.exists(os.path.join(save_dir, "best_model.pth")):
        import time, shutil
        backup_dir = f"{save_dir}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(save_dir, backup_dir)
        print(f"[overwrite guard] existing run found — archived to:\n  {backup_dir}")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    # ── datasets ─────────────────────────────────────────────────────────────
    train_ds = CloudSEN12Dataset(
        split="Train",
        terrain_mode=CONFIG["terrain_mode"],
        augment=CONFIG["augment"]
    )
    val_ds = CloudSEN12Dataset(
        split="Val",
        terrain_mode=CONFIG["terrain_mode"],
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,           # smaller batch for 512×512 val patches
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    # ── model ────────────────────────────────────────────────────────────────
    # dataset_channels = what the DataLoader yields; for "film" the encoder sees
    # 2 fewer (the lat/lon planes are peeled off and routed to the FiLM MLP).
    dataset_channels = IN_CHANNELS[CONFIG["terrain_mode"]]
    arch = CONFIG.get("arch", "unet")
    if arch == "tcm":
        # Stage 6: `elev` mode is 14ch = 13 bands + 1 elevation plane, so the
        # encoder takes ONE fewer channel (the lat/lon FiLM variants peel off
        # TWO). Elevation is routed to the cross-attention terrain branch.
        encoder_channels = dataset_channels - 1
        model = build_tcm_unet(
            in_channels  = encoder_channels,
            num_classes  = 4,
            encoder_name = "resnet34",
            pretrained   = True,
        ).to(device)
        print(f"Model  : TCM-UNet ResNet-34 | encoder_in={encoder_channels} "
              f"(+elevation via cross-attention @ feats[4],feats[5]) "
              f"| dataset_ch={dataset_channels}")
    elif arch == "gap":
        # gap_terrain ablation: TCM with attention deleted (the analytic
        # uniform-attention limit). Same 14ch input and same channel split as
        # arch="tcm", so it is a drop-in A/B on the injection mechanism.
        encoder_channels = dataset_channels - 1
        model = build_gap_terrain_unet(
            in_channels  = encoder_channels,
            num_classes  = 4,
            encoder_name = "resnet34",
            pretrained   = True,
        ).to(device)
        print(f"Model  : GAP-Terrain UNet ResNet-34 | encoder_in={encoder_channels} "
              f"(+elevation via global-avg-pooled terrain @ feats[4],feats[5]) "
              f"| dataset_ch={dataset_channels}")
    elif arch == "film_elev":
        # Pooled-elevation FiLM control: same 14ch elev_pooled tensor as the
        # concatenated null, but the patch-mean scalar is peeled off and
        # injected as gamma/beta at the bottleneck instead of entering the
        # encoder as a channel. ONE scalar, so -1 not -2.
        encoder_channels = dataset_channels - 1
        from model import build_film_elev_unet
        model = build_film_elev_unet(
            in_channels  = encoder_channels,
            num_classes  = 4,
            encoder_name = "resnet34",
            pretrained   = True,
        ).to(device)
        print(f"Model  : FiLM-Elev UNet ResNet-34 | encoder_in={encoder_channels} "
              f"(+patch-mean elevation via bottleneck FiLM) "
              f"| dataset_ch={dataset_channels}")
    elif arch == "gap_latlon":
        # gap + lat/lon FiLM: 16ch elev_latlon tensor; encoder takes the 13 bands,
        # elevation -> gap branch, lat/lon -> bottleneck FiLM. THREE planes peeled.
        encoder_channels = dataset_channels - 3
        from model import build_gap_latlon_unet
        model = build_gap_latlon_unet(
            in_channels  = encoder_channels,
            num_classes  = 4,
            encoder_name = "resnet34",
            pretrained   = True,
        ).to(device)
        print(f"Model  : GAP-Terrain + lat/lon FiLM UNet ResNet-34 | encoder_in={encoder_channels} "
              f"(+elevation via gap @ feats[4],feats[5]; +lat/lon via bottleneck FiLM) "
              f"| dataset_ch={dataset_channels}")
    elif arch in ("film", "film_multi"):
        encoder_channels = dataset_channels - 2
        # film_multi = Stage 5b(3): condition every encoder scale, not just the
        # bottleneck, so skip connections also carry geo signal.
        builder = build_multiscale_film_unet if arch == "film_multi" else build_film_unet
        model = builder(
            in_channels  = encoder_channels,
            num_classes  = 4,
            encoder_name = "resnet34",
            pretrained   = True,
        ).to(device)
        where = "all encoder scales" if arch == "film_multi" else "bottleneck"
        print(f"Model  : FiLM-UNet ResNet-34 | encoder_in={encoder_channels} "
              f"(+lat/lon via {where}) | dataset_ch={dataset_channels}")
    else:
        model = smp.Unet(
            encoder_name   = "resnet34",
            encoder_weights= "imagenet",   # pretrained encoder
            in_channels    = dataset_channels,
            classes        = 4,
        ).to(device)
        print(f"Model  : U-Net ResNet-34 | in_channels={dataset_channels}")

    # ── loss & optimizer ─────────────────────────────────────────────────────
    # NOTE: keeping plain Adam (not AdamW) deliberately, to stay consistent
    # with Stage 1 and Stage 2 so any mIoU change here can be attributed to
    # slope alone, not a confounded optimizer change. AdamW switch is planned
    # as a separate, isolated ablation after the full terrain chain completes.
    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )

    # ── resume support ───────────────────────
    # Guards against losing a long run to an external kill (5b(1) died at
    # ep57/100 when the machine slept). A full-state `last_checkpoint.pth` is
    # written every epoch; CONFIG["resume"]=True restores model + optimizer +
    # ReduceLROnPlateau state (so plateau/LR counters continue correctly) and
    # picks up at the next epoch. Restoring these states makes a resumed run
    # equivalent to an uninterrupted one — it does NOT change the training math,
    # so the ablation stays valid.
    best_miou   = 0.0
    history     = []
    start_epoch = 1

    last_ckpt_path = os.path.join(save_dir, "last_checkpoint.pth")
    if CONFIG.get("resume") and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_miou   = ckpt["best_miou"]
        start_epoch = ckpt["epoch"] + 1
        hist_path = os.path.join(save_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = [h for h in json.load(f) if h["epoch"] < start_epoch]
        print(f"RESUMED from epoch {ckpt['epoch']} "
              f"(best_miou={best_miou:.4f}) — continuing at epoch {start_epoch}")
    elif CONFIG.get("resume"):
        print("resume=True but no last_checkpoint.pth found — starting fresh")

    # ── training loop ────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, CONFIG["epochs"] + 1):
            print(f"\nEpoch {epoch}/{CONFIG['epochs']}")

            train_loss            = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_metrics = val_epoch(model, val_loader, criterion, device)

            miou = val_metrics["miou"]
            mf1  = val_metrics["mf1"]

            print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
            print(f"  mIoU={miou:.4f} | mF1={mf1:.4f}")
            print(f"  per-class IoU: clear={val_metrics['iou'][0]:.4f} "
                  f"thick={val_metrics['iou'][1]:.4f} "
                  f"thin={val_metrics['iou'][2]:.4f} "
                  f"shadow={val_metrics['iou'][3]:.4f}")

            # Stage 5b(2) diagnostic: how much is the conditioning actually doing?
            # Both start at exactly 0 (zero-init). If they stay ~0 the model is
            # CHOOSING to ignore lat/lon — which turns an ambiguous null into a
            # readable one.
            film_stats = {}
            if CONFIG.get("arch") in ("film", "gap_latlon"):   # gap_latlon has the same model.film + lat/lon val_ds
                with torch.no_grad():
                    ll = torch.tensor(
                        np.stack([val_ds.lats / 90.0, val_ds.lons / 180.0], axis=1),
                        dtype=torch.float32, device=device)
                    g, b = model.film.mlp(ll).chunk(2, dim=1)
                    film_stats = {
                        "film_gamma_absmean": g.abs().mean().item(),
                        "film_beta_absmean" : b.abs().mean().item(),
                    }
                print(f"  FiLM: |gamma|avg={film_stats['film_gamma_absmean']:.4e} "
                      f"| |beta|avg={film_stats['film_beta_absmean']:.4e}")

            # save best model
            if miou > best_miou:
                best_miou = miou
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_miou": best_miou,
                }
                atomic_torch_save(checkpoint, os.path.join(save_dir, "best_model.pth"))   # kill-safe write
                print(f"  *** new best mIoU: {best_miou:.4f} — model saved ***")

            # step scheduler ONCE, capture LR change
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(miou)
            lr_after = optimizer.param_groups[0]['lr']

            if lr_after != lr_before:
                print(f"  *** LR dropped: {lr_before:.2e} -> {lr_after:.2e} ***")

            # log history (lr included directly here, no separate epoch_record)
            history.append({
                "epoch"     : epoch,
                "train_loss": train_loss,
                "val_loss"  : val_loss,
                "miou"      : miou,
                "mf1"       : mf1,
                "iou_clear" : val_metrics["iou"][0],
                "iou_thick" : val_metrics["iou"][1],
                "iou_thin"  : val_metrics["iou"][2],
                "iou_shadow": val_metrics["iou"][3],
                "lr"        : lr_after,
                **film_stats,   # gamma/beta magnitudes, film arch only
            })

            # save history every epoch so you don't lose it on crash
            hist_path = os.path.join(save_dir, "history.json")   # tmp + os.replace so a kill mid-write cannot truncate it
            with open(hist_path + ".tmp", "w") as f:
                json.dump(history, f, indent=2)
            os.replace(hist_path + ".tmp", hist_path)

            # full-state checkpoint every epoch → resumable after any kill
            atomic_torch_save({   # kill-safe write: the watchdog auto-resumes from this file
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_miou": best_miou,
            }, last_ckpt_path)

    except KeyboardInterrupt:
        # Spyder "stop" / Ctrl-C: the last COMPLETED epoch is already saved to
        # last_checkpoint.pth, so just report and exit cleanly, no lost state.
        print(f"\nInterrupted. Last completed epoch is checkpointed to "
              f"{last_ckpt_path}. Set CONFIG['resume']=True and re-run to continue.")

    print(f"\nTraining complete (or stopped). Best mIoU so far: {best_miou:.4f}")
    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    main()