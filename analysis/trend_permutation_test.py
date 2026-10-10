"""
Pre-planned exact permutation tests on the paper's three primary claims.


Why: project notes "Statistics" item 1 (decided 2026-08-24) — test aggregate mIoU,
the slope trend and the altitude trend as ONE test each (not six per-bin tests),
with exact permutation p-values and Holm correction, under BOTH checkpoint
protocols (best-val and final-epoch).

Test unit = a training seed. For each comparison A vs B and each protocol:
  aggregate : statistic = mean_A(mIoU) - mean_B(mIoU)
  trend     : per bin k, d_k = mean_A(mIoU_k) - mean_B(mIoU_k); statistic = OLS
              slope of d_k on bin index k (equal weight per bin). A positive slope
              means A's advantage over B grows with slope / altitude.
p = fraction of all C(nA+nB, nA) relabelings of seeds into groups whose |statistic|
is >= the observed |statistic| (two-sided, exact). Holm across the 3 claims
within each comparison x protocol.

Bins: slope <2, 2-5, 5-10, 10-20, 20-30, >30 deg (v2 slope; joint_stratification
JSONs). Altitude 0-500, 500-1000, 1000-2000, 2000-3000, >3000 m (the <0 band is
0.5% of px with ~no thin cloud and is excluded as degenerate, as everywhere else).

No GPU: reads the stored confusion matrices only.
Output: experiments/stratified_comparison/trend_tests.json
"""
import itertools, json, os
import numpy as np

EXP = r"C:\Users\prash\toposegnet\experiments"

GROUPS = {
    "1a":        ["1a baseline (13ch)", "1a seed1", "1a seed2"],
    "concat":    ["2a concat seed1", "2a concat seed2", "2a concat seed3", "2a concat seed4"],
    "film_elev": ["film_elev seed0", "film_elev seed1", "film_elev seed2", "film_elev seed3"],
    "TCM":       ["6 elev CROSS-ATTN (TCM)", "6 TCM seed1", "6 TCM seed2"],
    "gap":       ["gap seed0", "gap seed1", "gap seed2", "gap seed3"],
}
# primary comparisons first; the others are reported as exploratory
COMPARISONS = [("gap", "concat", "primary"), ("concat", "1a", "primary"),
               ("film_elev", "concat", "exploratory"), ("TCM", "concat", "exploratory"),
               ("gap", "1a", "exploratory")]
ALT_BANDS = ["0-500", "500-1000", "1000-2000", "2000-3000", ">3000"]


def miou(conf):
    c = np.asarray(conf, float)
    inter = np.diag(c)
    union = c.sum(0) + c.sum(1) - inter
    return float(np.nanmean(np.where(union > 0, inter / np.maximum(union, 1), np.nan)))


def ols_slope(d):
    x = np.arange(len(d), dtype=float)
    x -= x.mean()
    return float((x * (np.asarray(d) - np.mean(d))).sum() / (x * x).sum())


def exact_perm(a_rows, b_rows, stat):
    """a_rows/b_rows: per-seed feature vectors. stat(A, B) -> scalar. Two-sided exact p."""
    rows = a_rows + b_rows
    na, n = len(a_rows), len(rows)
    obs = stat(a_rows, b_rows)
    hits = total = 0
    for idx in itertools.combinations(range(n), na):
        s = set(idx)
        A = [rows[i] for i in idx]
        B = [rows[i] for i in range(n) if i not in s]
        hits += abs(stat(A, B)) >= abs(obs) - 1e-12
        total += 1
    return obs, hits / total, total


def agg_stat(A, B):
    return np.mean([r[0] for r in A]) - np.mean([r[0] for r in B])


def trend_stat(A, B):
    return ols_slope(np.mean(A, axis=0) - np.mean(B, axis=0))


def holm(ps):
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    out = {"bins": {"slope": None, "altitude": ALT_BANDS}, "results": {}}
    for proto, sfx in [("best_val", ""), ("final_epoch", "_final")]:
        S = json.load(open(os.path.join(EXP, "stratified_comparison", f"stratified_comparison{sfx}.json")))
        J = json.load(open(os.path.join(EXP, "joint_stratification", f"joint_stratification{sfx}.json")))
        out["bins"]["slope"] = J["slope_labels"]
        ek = [S["elev_labels"].index(b) for b in ALT_BANDS]

        def feats(name):
            m, j = S["models"].get(name), J["models"].get(name)
            if m is None or j is None:
                return None
            return {"agg": [miou(m["conf_all"])],
                    "slope": [miou(c) for c in j["conf_slope"]],
                    "alt": [miou(m["conf_elev"][k]) for k in ek]}

        F = {g: [f for f in (feats(n) for n in ns) if f is not None] for g, ns in GROUPS.items()}
        res = {}
        print(f"\n=== {proto} ===  n: " + ", ".join(f"{g}={len(v)}" for g, v in F.items()))
        for a, b, tier in COMPARISONS:
            A, B = F[a], F[b]
            claims = {}
            for claim, key, stat in [("aggregate", "agg", agg_stat),
                                     ("slope_trend", "slope", trend_stat),
                                     ("altitude_trend", "alt", trend_stat)]:
                obs, p, nperm = exact_perm([r[key] for r in A], [r[key] for r in B], stat)
                d = (np.mean([r[key] for r in A], 0) - np.mean([r[key] for r in B], 0)).tolist()
                claims[claim] = {"statistic": obs, "p": p, "n_relabelings": nperm,
                                 "per_bin_delta": d if key != "agg" else None}
            for c, adj in zip(claims, holm([claims[c]["p"] for c in claims])):
                claims[c]["p_holm"] = adj
            res[f"{a}_vs_{b}"] = {"tier": tier, "nA": len(A), "nB": len(B), "claims": claims}
            print(f"  {a:9s} vs {b:6s} [{tier}] (n={len(A)},{len(B)}; {claims['aggregate']['n_relabelings']} relabelings)")
            for c, r in claims.items():
                unit = "mIoU" if c == "aggregate" else "mIoU/bin"
                print(f"     {c:15s} {r['statistic']:+.5f} {unit:9s} p={r['p']:.3f}  Holm={r['p_holm']:.3f}")
        out["results"][proto] = res
    dst = os.path.join(EXP, "stratified_comparison", "trend_tests.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
