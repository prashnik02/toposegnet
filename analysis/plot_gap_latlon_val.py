"""
Val mIoU curves: gap_latlon (in progress) vs the three completed gap seeds.


Re-run any time while the job trains — it reads history.json, which train.py
rewrites every epoch. Output: experiments/gap_latlon_elev_latlon_augFalse/val_miou_curves.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = r"C:\Users\prash\toposegnet\experiments"
RUN = "gap_latlon_elev_latlon_augFalse"
GAP = ["gap_elev_augFalse", "gap_elev_augFalse_seed1", "gap_elev_augFalse_seed2"]

BLUE, GRAY, INK, INK2, GRID = "#2a78d6", "#9a9994", "#0b0b0b", "#52514e", "#e6e5e1"   # reference palette slot 1 + neutrals


def load(name):
    with open(os.path.join(EXP, name, "history.json")) as f:
        h = json.load(f)
    return [x["epoch"] for x in h], [x["miou"] for x in h], [x["lr"] for x in h]


fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

finals = []
for i, g in enumerate(GAP):
    ep, mi, _ = load(g)
    finals.append(max(mi))
    ax.plot(ep, mi, color=GRAY, lw=1.1, alpha=0.75, zorder=2,
            label="gap, seeds 0/1/2 (completed)" if i == 0 else None)

# gap's final best-val range — where a finished run is expected to land
ax.axhspan(min(finals), max(finals), color=GRAY, alpha=0.15, lw=0, zorder=1)
ax.text(101, (min(finals) + max(finals)) / 2, f"gap best-val\n{min(finals):.4f}–{max(finals):.4f}",
        va="center", ha="left", fontsize=8.5, color=INK2)

ep, mi, lr = load(RUN)
ax.plot(ep, mi, color=BLUE, lw=2.2, zorder=4, label="gap + lat/lon FiLM, seed 0 (training)")
b = max(range(len(mi)), key=mi.__getitem__)
ax.scatter([ep[b]], [mi[b]], s=46, color=BLUE, edgecolor="#fcfcfb", linewidth=2, zorder=5)
ax.annotate(f"best {mi[b]:.4f} (ep {ep[b]})", (ep[b], mi[b]), xytext=(8, 10),
            textcoords="offset points", fontsize=9, color=INK)
for k in range(1, len(lr)):   # mark LR drops on the new run
    if lr[k] < lr[k - 1]:
        ax.axvline(ep[k], color=BLUE, lw=0.8, ls=":", zorder=3)
ax.axvline(ep[-1], color=INK2, lw=0.8, ls="--", zorder=3)
ax.text(ep[-1] + 0.8, 0.585, f"now: ep {ep[-1]}/100",
        fontsize=8.5, color=INK2, va="bottom")

ax.set_xlim(0, 100); ax.set_ylim(0.58, 0.74)
ax.set_xlabel("epoch", color=INK2); ax.set_ylabel("validation mIoU", color=INK2)
ax.set_title("Validation mIoU — gap + lat/lon FiLM vs gap (n=3)", loc="left", color=INK, fontsize=12)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
fig.text(0.01, 0.01, "Val mIoU is noisy and has disagreed with test on this task; the decision metric is final TEST mIoU > 0.7036.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 0.93, 1))

out = os.path.join(EXP, RUN, "val_miou_curves.png")
fig.savefig(out, facecolor=fig.get_facecolor())
print(out)
