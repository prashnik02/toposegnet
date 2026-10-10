"""
Run training / evaluation / analysis steps back to back.
Extended 2026-10-05 with eval-only and script steps for Statistics items 2-5.

Safe to re-launch after ANY kill (sleep, app update, reboot) — every step
is idempotent:
  - train : skipped if history.json already reaches EPOCHS; RESUMED if a
            last_checkpoint.pth exists (full model+optimizer+scheduler restore,
            equivalent to uninterrupted); otherwise starts fresh. Followed by a
            best-val + final-epoch test eval.
  - eval  : skipped if the target test_results.json already exists.
  - script: the analysis scripts are themselves incremental (joint) or cache
            stored confusions (stratified), so re-running only redoes new models.

Each step runs in its own subprocess, so GPU memory and RNG state never leak
between runs. Hyperparameters are NOT touched — only terrain_mode/arch/seed,
as the one-variable-at-a-time chain requires.

Usage (repo root, toposegnet env python):
    python run_queue.py                   # run now
    python run_queue.py --after-pid 1234  # wait for another queue to exit first
    python run_queue.py --final-strat     # stratify last_checkpoint.pth instead
"""
import json, os, subprocess, sys, time

ROOT = r"C:\Users\prash\toposegnet"
EXP  = os.path.join(ROOT, "experiments")
PY   = sys.executable
EPOCHS = 100


def T(terrain_mode, arch, seed):          # train (+ best-val and final-epoch eval)
    return dict(kind="train", terrain_mode=terrain_mode, arch=arch, seed=seed)


def E(dirname, terrain_mode, arch, which):  # eval only; which = "best" | "final"
    return dict(kind="eval", dir=dirname, terrain_mode=terrain_mode, arch=arch, which=which)


def S(path, env=None):                    # analysis script (+ optional extra env vars)
    return dict(kind="script", path=path, env=env or {})


# Final-epoch re-eval targets (Statistics item 2): the five n=3 configurations.
# The seedless original 1a has no last_checkpoint.pth (predates resume support),
# so it cannot get a final-epoch number.
N3_RUNS = (
    [(f"unet_elev_augFalse_seed{s}", "elev", "unet") for s in (1, 2, 3)]
    + [(f"tcm_elev_augFalse{sfx}", "elev", "tcm") for sfx in ("", "_seed1", "_seed2")]
    + [(f"gap_elev_augFalse{sfx}", "elev", "gap") for sfx in ("", "_seed1", "_seed2")]
    + [(f"unet_elev_pooled_augFalse{sfx}", "elev_pooled", "unet") for sfx in ("", "_seed1", "_seed2")]
    + [(f"film_elev_elev_pooled_augFalse{sfx}", "elev_pooled", "film_elev") for sfx in ("", "_seed1", "_seed2")]
)

STEPS = (
    # 1a seeds (already run by the 2026-10-03 queue; listed so a relaunch is a no-op)
    [T("none", "unet", 1), T("none", "unet", 2)]
    # item 2 — original seedless 1a, best-val only, through the current evaluator
    + [E("unet_none_augFalse", "none", "unet", "best")]
    # item 3 — final-epoch re-eval of every n=3 run
    + [E(d, m, a, "final") for d, m, a in N3_RUNS]
    # item 4 — put the 1a seeds into the altitude/surface and slope stratifications
    + [S(r"analysis\stratified_model_comparison.py"),
       S(r"analysis\joint_slope_elev_stratification.py")]
    # item 5 — 4th replicate of the three primary configs (concat already has 1/2/3)
    + [T("elev", "unet", 4)]
    # final-epoch stratification moved up to run right after concat seed 4, at the
    # user's request, so the protocol-robustness answer does not wait ~2.5 days for
    # the remaining seed runs
    + [S(r"analysis\stratified_model_comparison.py", {"TOPOSEG_STRAT_CKPT": "final"}),
       S(r"analysis\joint_slope_elev_stratification.py", {"TOPOSEG_STRAT_CKPT": "final"})]
    + [T("elev_pooled", "film_elev", 3), T("elev", "gap", 3)]
    # ... and stratify them, best-val then final-epoch (both cache/skip models already done)
    + [S(r"analysis\stratified_model_comparison.py"),
       S(r"analysis\joint_slope_elev_stratification.py"),
       S(r"analysis\stratified_model_comparison.py", {"TOPOSEG_STRAT_CKPT": "final"}),
       S(r"analysis\joint_slope_elev_stratification.py", {"TOPOSEG_STRAT_CKPT": "final"})]
    # ── Spectral-ablation follow-up, Phase 1 n=2 screen (project notes ACTIVE PLAN, 2026-10-10).
    # Interleaved so a first none_4 / gap_4 / concat_4 look exists after 3 runs.
    # Hyperparameters identical to the chain; only terrain_mode / arch / seed differ.
    # Everything above is already finished, so on relaunch those steps are no-ops.
    + [T("rgbn", "unet", 0), T("rgbn_elev", "gap", 0), T("rgbn_elev", "unet", 0),
       T("rgbn", "unet", 1), T("rgbn_elev", "gap", 1), T("rgbn_elev", "unet", 1)]
    + [S(r"analysis\stratified_model_comparison.py"),
       S(r"analysis\joint_slope_elev_stratification.py"),
       S(r"analysis\stratified_model_comparison.py", {"TOPOSEG_STRAT_CKPT": "final"}),
       S(r"analysis\joint_slope_elev_stratification.py", {"TOPOSEG_STRAT_CKPT": "final"}),
       S(r"analysis\spectral_screen.py")]
)


# Final-epoch stratification (Statistics item 2, stratified): same two scripts, but
# TOPOSEG_STRAT_CKPT=final makes them load last_checkpoint.pth and write *_final.json.
# Run with --final-strat. The seedless 1a has no last_checkpoint.pth and is skipped.
FINAL_STRAT_STEPS = [
    S(r"analysis\stratified_model_comparison.py", {"TOPOSEG_STRAT_CKPT": "final"}),
    S(r"analysis\joint_slope_elev_stratification.py", {"TOPOSEG_STRAT_CKPT": "final"}),
]


def folder(job):
    # mirrors train.py's naming: suffix only when seed != 0
    name = f"{job['arch']}_{job['terrain_mode']}_augFalse"
    if job["seed"]:
        name += f"_seed{job['seed']}"
    return os.path.join(EXP, name)


def last_epoch(d):
    p = os.path.join(d, "history.json")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        h = json.load(f)
    return max((x["epoch"] for x in h), default=0)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def run_eval(d, terrain_mode, arch, which):
    """One test eval; skipped if its results file already exists."""
    if which == "best":
        ckpt, save = os.path.join(d, "best_model.pth"), d
    else:
        ckpt, save = os.path.join(d, "last_checkpoint.pth"), os.path.join(d, "final_epoch_eval")
    res = os.path.join(save, "test_results.json")
    tag = f"{os.path.basename(d)} [{which}]"
    if os.path.exists(res):
        log(f"SKIP eval {tag} — results exist")
        return
    if not os.path.exists(ckpt):
        log(f"SKIP eval {tag} — no {os.path.basename(ckpt)}")
        return
    log(f"EVAL {tag}")
    code = (
        "import sys; sys.path.append(r'%s\\src')\n"
        "from evaluate_notebook import run_test_eval\n"
        "run_test_eval('%s', r'%s', save_dir=r'%s', arch='%s')\n"
    ) % (ROOT, terrain_mode, ckpt, save, arch)
    subprocess.call([PY, "-u", "-c", code], cwd=ROOT,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    if os.path.exists(res):
        with open(res) as f:
            log(f"  {tag}: test mIoU {json.load(f)['test_miou']:.5f}")
    else:
        log(f"  {tag}: FAILED — no results written")


def run_train(job):
    d = folder(job)
    done = last_epoch(d)
    if done >= EPOCHS:
        log(f"SKIP training {os.path.basename(d)} — already at epoch {done}")
    else:
        resume = done > 0 and os.path.exists(os.path.join(d, "last_checkpoint.pth"))
        override = dict(terrain_mode=job["terrain_mode"], arch=job["arch"],
                        seed=job["seed"], resume=resume)
        log(f"TRAIN {os.path.basename(d)} | resume={resume} (history at ep {done})")
        env = dict(os.environ, TOPOSEG_CONFIG_OVERRIDE=json.dumps(override),
                   PYTHONIOENCODING="utf-8")
        rc = subprocess.call([PY, "-u", "train.py"], cwd=ROOT, env=env)
        done = last_epoch(d)
        log(f"train.py exited rc={rc}, history now at ep {done}")
        if done < EPOCHS:
            return False
    run_eval(d, job["terrain_mode"], job["arch"], "best")
    run_eval(d, job["terrain_mode"], job["arch"], "final")
    return True


def run_script(path, extra_env=None):
    log(f"SCRIPT {path} {extra_env or ''}")
    rc = subprocess.call([PY, "-u", path], cwd=ROOT,
                         env=dict(os.environ, PYTHONIOENCODING="utf-8", **(extra_env or {})))
    log(f"  {path} exited rc={rc}")
    return rc == 0


def acquire_single_instance_lock():
    """Exit immediately if another run_queue.py already holds the lock.

    The watchdog (watchdog.ps1) and a human can both start the queue; two copies
    would resume the same run into the same folder. msvcrt.locking takes an OS
    byte-range lock that Windows drops when the process dies, so a crashed queue
    can never leave a stale lock behind.
    """
    import msvcrt
    os.makedirs(os.path.join(EXP, "_logs"), exist_ok=True)
    fh = open(os.path.join(EXP, "_logs", "run_queue.lock"), "a+")
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log("another run_queue.py holds the lock — exiting")
        sys.exit(0)
    return fh   # keep the handle open for the life of the process


if __name__ == "__main__":
    _lock = acquire_single_instance_lock()   # one queue at a time
    if "--after-pid" in sys.argv:
        pid = int(sys.argv[sys.argv.index("--after-pid") + 1])
        log(f"waiting for PID {pid} to exit")
        while pid_alive(pid):
            time.sleep(60)
        log(f"PID {pid} gone — starting")

    steps = FINAL_STRAT_STEPS if "--final-strat" in sys.argv else STEPS
    for step in steps:
        if step["kind"] == "train":
            if not run_train(step):
                log(f"STOP — {os.path.basename(folder(step))} did not reach epoch {EPOCHS}; "
                    f"re-run run_queue.py to resume")
                sys.exit(1)
        elif step["kind"] == "eval":
            run_eval(os.path.join(EXP, step["dir"]), step["terrain_mode"],
                     step["arch"], step["which"])
        elif step["kind"] == "script":
            if not run_script(step["path"], step.get("env")):
                log("STOP — analysis script failed; fix and re-run run_queue.py")
                sys.exit(1)
    log("QUEUE COMPLETE")
