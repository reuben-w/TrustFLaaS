r"""
Final-battery launcher. NEW file: changes nothing else.

Each dataset takes ALL its settings (split, minority class, rounds, learning rates, DALSE settings, ...)
from one saved *_config.json of a finished run of that dataset. The arm then overrides only what defines it.

Arms (all with smashed-poison attack unless noted):
  A  HealSplit baseline          detector=tas          healing=gan    no LA loss
  B  new detector + GAN          detector=client_shift healing=gan    no LA loss
  C  TrustFLS                    detector=client_shift healing=dalse  LA loss psi=1
  D  ablation: drop              as C, flagged clients dropped instead of healed        (seed 42 only)
  E  TrustFLS, no attack         as C with --no-attack (cost of the defence when nothing is wrong)
  U  undefended, attacked        no defence at all (re-run on GPU so every row shares hardware)
  V  undefended, clean           no defence, no attack
Technique comparison (all use the new client-shift detector, so only the healing / class-aware part differs):
  F  BAGAN-GP healing            healing=bagan_gp (Huang & Jafari), no class-aware loss
  G  DALSE + class-balanced loss cb_loss_beta=0.999 (Cui et al. 2019)
  H  DALSE + LA + EWC            la psi=1, ewc lambda=10 from round 5 (MConFL, Yuan et al. 2026)
  I  DALSE + LA + contrastive    needs --contrastive-lambda (supervised contrastive refinement, Khosla et al.)
  J  DALSE, no class-aware loss  la psi=0 (isolates what the logit-adjusted loss adds to C)

Order: seed by seed (all datasets, all arms) so a partial run is still a usable, balanced result.
Resumable: a run whose metrics file already has every round is skipped. Run from the folder that holds
sfl_healsplit\ and src\ :
  python -m sfl_healsplit.run_battery --configs <mnist cfg> <fmnist cfg> <cifar cfg> <pamap2 cfg> --out final_results --device cuda
Add --dry-run to just print the commands.
"""
import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
import time

from .config import SFLConfig

ARMS = {
    "A": dict(detector="tas", healing_mode="gan", la_loss_psi=0.0, attack=True),
    "B": dict(detector="client_shift", healing_mode="gan", la_loss_psi=0.0, attack=True),
    "C": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=1.0, attack=True),
    "D": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=1.0, attack=True,
              flagged_client_action="drop"),
    "E": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=1.0, attack=False),
    "F": dict(detector="client_shift", healing_mode="bagan_gp", la_loss_psi=0.0, attack=True),
    "G": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=0.0, cb_loss_beta=0.999, attack=True),
    "H": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=1.0, ewc_lambda=10.0,
              ewc_start_round=5, attack=True),
    "I": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=1.0, attack=True),   # + contrastive
    "J": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=0.0, attack=True),
    "U": dict(defense=False, healing_mode="gan", la_loss_psi=0.0, attack=True),   # undefended, attacked
    "V": dict(defense=False, healing_mode="gan", la_loss_psi=0.0, attack=False),  # undefended, clean
    "M": dict(detector="client_shift", healing_mode="dalse", la_loss_psi=1.0, attack=False),   # arm I with no attack (+ contrastive)
}
ARM_SEEDS = {"D": [42]}                # ablation: one seed only, unless --seeds is narrower
# Fields every arm pins, so a stray technique setting in the source config cannot leak into an arm.
COMMON = dict(defense=True, attack_type="smashed_poison", cb_loss_beta=0.0, ewc_lambda=0.0,
              contrastive_lambda=0.0, flagged_client_action="heal")
FIELDS = {f.name: f for f in dataclasses.fields(SFLConfig)}
BOOLS = {"iid", "attack", "defense"}


def argv_from(cfg: dict):
    out = []
    for name in FIELDS:
        if name not in cfg:
            continue
        v = cfg[name]
        if name == "iid":
            out.append("--iid" if v else "--non-iid")
        elif name == "attack":
            out.append("--attack" if v else "--no-attack")
        elif name == "defense":
            out.append("--defense" if v else "--no-defense")
        else:
            out += ["--" + name.replace("_", "-"), str(v)]
    return out


def finished(metrics_path, num_rounds):
    try:
        with open(metrics_path, newline="") as f:
            return sum(1 for _ in csv.reader(f)) >= num_rounds + 2      # header + rounds 0..N
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", default="final_results")
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--arms", default="ABCDE")
    ap.add_argument("--contrastive-lambda", type=float, default=None,
                    help="strength for arm I (supervised contrastive refinement); arm I is skipped without it")
    ap.add_argument("--sigma", type=float, default=None,
                    help="override the smashed-poison strength (default: the config's, 3.0). Run names get "
                         "an _s<sigma> tag so they never overwrite the default-strength runs.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--datasets", nargs="*", default=None, help="only these dataset names")
    ap.add_argument("--split", default="config", choices=["config", "iid", "noniid"],
                    help="config = keep each dataset's saved split; iid / noniid = force it for every "
                         "dataset (non-IID uses the config's dirichlet_alpha). Run names carry '_noniid' "
                         "for non-IID runs, so IID and non-IID results never overwrite each other.")
    ap.add_argument("--num-rounds", type=int, default=None, help="override rounds (smoke tests only)")
    ap.add_argument("--la-counts", default="full", choices=["full", "kept"],
                    help="LA prior counts; kept = exclude dropped samples. Run names get '_lakept'.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = {}
    for p in args.configs:
        with open(p) as f:
            c = json.load(f)
        base[c["dataset"]] = c
    if args.datasets:
        base = {k: v for k, v in base.items() if k in args.datasets}

    jobs = []
    for seed in args.seeds:
        for ds, c0 in base.items():
            for arm in args.arms:
                if seed not in ARM_SEEDS.get(arm, args.seeds):
                    continue
                if arm in ("I", "M") and args.contrastive_lambda is None:
                    print(f"skipping arm {arm}: pass --contrastive-lambda <value>")
                    continue
                cfg = dict(c0)
                cfg.update(COMMON)
                cfg.update(ARMS[arm])
                if arm in ("I", "M"):
                    cfg["contrastive_lambda"] = args.contrastive_lambda
                if args.sigma is not None:
                    cfg["smashed_poison_sigma"] = args.sigma
                cfg.update(seed=seed, device=args.device, results_dir=os.path.join(args.out, ds))
                if args.split != "config":
                    cfg["iid"] = (args.split == "iid")
                if args.num_rounds:
                    cfg["num_rounds"] = args.num_rounds
                cfg["la_counts"] = args.la_counts
                sig = "" if args.sigma is None else f"_s{args.sigma:g}"
                cfg["run_name"] = f"{ds}_{arm}{'' if cfg['iid'] else '_noniid'}{sig}{'_lakept' if args.la_counts == 'kept' else ''}_seed{seed}"
                jobs.append((cfg, arm))

    print(f"{len(jobs)} runs planned")
    done = failed = 0
    t_all = time.time()
    for i, (cfg, arm) in enumerate(jobs, 1):
        run_dir = os.path.join(cfg["results_dir"], cfg["run_name"])
        metrics = os.path.join(run_dir, cfg["run_name"] + "_metrics.csv")
        cmd = [sys.executable, "-m", "sfl_healsplit.main"] + argv_from(cfg)
        if finished(metrics, cfg["num_rounds"]):
            print(f"[{i}/{len(jobs)}] skip (finished): {cfg['run_name']}")
            continue
        print(f"[{i}/{len(jobs)}] {cfg['run_name']}")
        if args.dry_run:
            print("   ", " ".join(f'"{c}"' if c == "" or " " in c else c for c in cmd))
            continue
        os.makedirs(cfg["results_dir"], exist_ok=True)
        log = os.path.join(cfg["results_dir"], cfg["run_name"] + "_console.txt")
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if rc == 0 and finished(metrics, cfg["num_rounds"]):
            done += 1
            print(f"    done in {mins:.1f} min")
        else:
            failed += 1
            print(f"    FAILED (exit {rc}) after {mins:.1f} min -- see {log}")
    print(f"finished: {done} run, {failed} failed, {(time.time() - t_all) / 3600:.2f} h")


if __name__ == "__main__":
    main()
