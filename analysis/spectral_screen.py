"""
Spectral-ablation Phase 1 screen: applies the PRE-REGISTERED GO / STOP / GREY rule.


Rule fixed in project notes (ACTIVE PLAN, "Pre-registered screening rule", 2026-10-10)
BEFORE any 4-band run — this script implements it verbatim; do not tune it to data.

Primary comparison gap_4 - none_4; strata overall and >3000 m (DEM-only);
protocols best-val (primary) and final-epoch.
  GO    : all gap_4 seeds above all none_4 seeds (overall OR >3000 m, best-val)
          AND mean delta at >3000 m >= +0.010 (best-val)
          AND the >3000 m delta is positive at final-epoch too.
  STOP  : mean delta <= +0.002 overall AND <= +0.002 at >3000 m, ranges overlapping,
          in both protocols.
  GREY  : anything else -> add seed 2 to all three configs and re-apply at n=3;
          still GREY at n=3 -> treat as STOP for escalation.
concat_4 is exploratory and never triggers GO. At n=2 v 2 the minimum exact p is
0.33 — this is a go/no-go on size + consistency, never a significance claim.

Inputs (no GPU): each run's test_results.json and final_epoch_eval/test_results.json
(overall mIoU) and stratified_comparison[_final].json (>3000 m mIoU).
Output: experiments/stratified_comparison/spectral_screen.json
Usage:  python analysis/spectral_screen.py            # real verdict
        python analysis/spectral_screen.py --selftest # check the rule logic on synthetic inputs
"""
import json, os, sys
import numpy as np

EXP = r"C:\Users\prash\toposegnet\experiments"
STRAT = os.path.join(EXP, "stratified_comparison")
GO_ALT_MIN, STOP_MAX = 0.010, 0.002

CONFIGS = {   # name in stratified_comparison MODELS -> run dir (seed 0, 1, then 2 if GREY)
    "none_4":   [("none_4 seed%d" % s,   "unet_rgbn_augFalse" + ("" if s == 0 else "_seed%d" % s)) for s in (0, 1, 2)],
    "gap_4":    [("gap_4 seed%d" % s,    "gap_rgbn_elev_augFalse" + ("" if s == 0 else "_seed%d" % s)) for s in (0, 1, 2)],
    "concat_4": [("concat_4 seed%d" % s, "unet_rgbn_elev_augFalse" + ("" if s == 0 else "_seed%d" % s)) for s in (0, 1, 2)],
}
REF_13 = {"1a": ["1a baseline (13ch)", "1a seed1", "1a seed2"],
          "gap": ["gap seed0", "gap seed1", "gap seed2", "gap seed3"]}


def miou(conf):
    c = np.asarray(conf, float); i = np.diag(c); u = c.sum(0) + c.sum(1) - i
    return float(np.nanmean(np.where(u > 0, i / np.maximum(u, 1), np.nan)))


def disjoint_above(a, b):
    return len(a) > 0 and len(b) > 0 and min(a) > max(b)


def overlap(a, b):
    return not (min(a) > max(b) or max(a) < min(b))


def verdict(vals):
    """vals[proto][stratum][config] -> list of per-seed mIoU. Returns (verdict, details)."""
    g, n = "gap_4", "none_4"
    d = {p: {s: float(np.mean(vals[p][s][g]) - np.mean(vals[p][s][n])) for s in ("overall", ">3000")}
         for p in ("best_val", "final_epoch")}
    sep = {s: disjoint_above(vals["best_val"][s][g], vals["best_val"][s][n]) for s in ("overall", ">3000")}
    go = ((sep["overall"] or sep[">3000"])
          and d["best_val"][">3000"] >= GO_ALT_MIN
          and d["final_epoch"][">3000"] > 0)
    stop = all(d[p]["overall"] <= STOP_MAX and d[p][">3000"] <= STOP_MAX
               and overlap(vals[p]["overall"][g], vals[p]["overall"][n])
               and overlap(vals[p][">3000"][g], vals[p][">3000"][n])
               for p in ("best_val", "final_epoch"))
    v = "GO" if go else ("STOP" if stop else "GREY")
    return v, {"delta": d, "best_val_all_gap_above_all_none": sep,
               "go_conditions": {"seed_separation_overall_or_3000": sep["overall"] or sep[">3000"],
                                 "best_val_3000_delta_ge_0.010": d["best_val"][">3000"] >= GO_ALT_MIN,
                                 "final_3000_delta_positive": d["final_epoch"][">3000"] > 0}}


def load(proto):
    sj = os.path.join(STRAT, "stratified_comparison" + ("" if proto == "best_val" else "_final") + ".json")
    S = json.load(open(sj)) if os.path.exists(sj) else {"models": {}, "elev_labels": []}
    k = S["elev_labels"].index(">3000") if ">3000" in S["elev_labels"] else None
    out = {"overall": {}, ">3000": {}}
    for cfg, runs in {**CONFIGS, **{c: [(m, None) for m in ms] for c, ms in REF_13.items()}}.items():
        ov, alt = [], []
        for name, d in runs:
            if d is not None:
                tr = os.path.join(EXP, d, "test_results.json" if proto == "best_val" else os.path.join("final_epoch_eval", "test_results.json"))
                m = S["models"].get(name)
                if not (os.path.exists(tr) and m is not None):
                    continue
                ov.append(json.load(open(tr))["test_miou"])
            else:
                m = S["models"].get(name)
                if m is None:
                    continue
                ov.append(miou(m["conf_all"]))
            alt.append(miou(m["conf_elev"][k]))
        out["overall"][cfg], out[">3000"][cfg] = ov, alt
    return out


def main():
    vals = {p: load(p) for p in ("best_val", "final_epoch")}
    ns = {p: {c: len(vals[p]["overall"][c]) for c in CONFIGS} for p in vals}
    print("Spectral-ablation screen (pre-registered rule, project notes 2026-10-10)")
    for p in vals:
        print(f"\n  {p}: n = " + ", ".join(f"{c}={ns[p][c]}" for c in CONFIGS))
        for s in ("overall", ">3000"):
            for c in ("none_4", "gap_4", "concat_4", "1a", "gap"):
                v = vals[p][s][c]
                if v:
                    print(f"    {s:8s} {c:9s} {np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]  n={len(v)}")
    n_ok = all(ns[p]["none_4"] >= 2 and ns[p]["gap_4"] >= 2 and ns[p]["none_4"] == ns[p]["gap_4"] for p in vals)
    result = {"n": ns, "values": vals}
    if not n_ok:
        result["verdict"] = "INCOMPLETE"
        print("\nVERDICT: INCOMPLETE — need >=2 (and equal) completed none_4 and gap_4 seeds in BOTH protocols.")
    else:
        v, det = verdict(vals)
        if v == "GREY" and ns["best_val"]["gap_4"] >= 3:
            v = "GREY_AT_N3 -> treat as STOP for escalation"
        result.update({"verdict": v, **det})
        print("\n  gap_4 - none_4 mean delta: " + json.dumps(det["delta"]))
        print("  GO conditions: " + json.dumps(det["go_conditions"]))
        print(f"\nVERDICT: {v}")
        if "concat_4" in vals["best_val"]["overall"] and vals["best_val"]["overall"]["concat_4"]:
            dc = {p: {s: float(np.mean(vals[p][s]["concat_4"]) - np.mean(vals[p][s]["none_4"]))
                      for s in ("overall", ">3000")} for p in vals if vals[p]["overall"]["concat_4"]}
            result["exploratory_concat4_minus_none4"] = dc
            print("  (exploratory) concat_4 - none_4: " + json.dumps(dc))
    os.makedirs(STRAT, exist_ok=True)
    with open(os.path.join(STRAT, "spectral_screen.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {os.path.join(STRAT, 'spectral_screen.json')}")


def selftest():
    def mk(none_bv, gap_bv, none_fe, gap_fe):   # each: (overall list, >3000 list)
        return {"best_val": {"overall": {"none_4": none_bv[0], "gap_4": gap_bv[0]}, ">3000": {"none_4": none_bv[1], "gap_4": gap_bv[1]}},
                "final_epoch": {"overall": {"none_4": none_fe[0], "gap_4": gap_fe[0]}, ">3000": {"none_4": none_fe[1], "gap_4": gap_fe[1]}}}
    go = mk(([.60, .61], [.50, .51]), ([.62, .63], [.53, .54]), ([.60, .61], [.50, .51]), ([.61, .62], [.51, .52]))
    stop = mk(([.60, .61], [.50, .51]), ([.601, .611], [.501, .508]), ([.60, .61], [.50, .51]), ([.601, .611], [.500, .510]))
    grey_small = mk(([.60, .61], [.50, .51]), ([.62, .63], [.505, .512]), ([.60, .61], [.50, .51]), ([.61, .62], [.501, .511]))
    grey_finalneg = mk(([.60, .61], [.50, .51]), ([.62, .63], [.53, .54]), ([.60, .61], [.50, .51]), ([.61, .62], [.48, .49]))
    for name, case, want in [("go", go, "GO"), ("stop", stop, "STOP"),
                             ("grey: >3000 delta below +0.010", grey_small, "GREY"),
                             ("grey: final >3000 negative", grey_finalneg, "GREY")]:
        got = verdict(case)[0]
        assert got == want, (name, got, want)
        print(f"  selftest {name:32s} -> {got}  OK")
    print("rule logic OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
