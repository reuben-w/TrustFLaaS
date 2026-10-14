r"""
Minority-subsampling sweep (supervisor request, 12 Oct 2026). NEW file: changes nothing else.
Put it in sfl_healsplit\ next to run_battery.py and run_retest.py (it needs both).

Trains each method on training data where the scarce class is kept at a given fraction of its original size
(0.5 = keep 50%, 0.7 = keep 70%, 1.0 = no subsampling), IID partition only, no attacker, then tests as usual
on the FULL unsubsampled test set. The minority-class-only test is the minority-class recall, which every run
already saves in its confusion matrix (retest_keepfrac.py reads it).

Methods: V plain split FL, N LA loss, O CB loss, P LA + EWC, M TrustFLaaS (arm M, needs --contrastive-lambda).
The 20% runs already exist (V, M in final_results; N, O, P in final_results\retest) and are NOT rerun.
PAMAP2 is left out: its scarce class is naturally rare and --minority-keep-fraction is a no-op for it.

Defaults: fractions 0.5 0.7 1.0, datasets mnist fashion-mnist cifar10, seeds 42-46 -> 3 x 3 x 5 x 5 = 225 runs.
Output: final_results\keepfrac\k50\..., k70\..., k100\...  (one folder per fraction, so it zips on its own).
Finished runs are skipped, so it can be stopped and resumed. Runs go seed by seed, so a partial result is balanced.

  python -m sfl_healsplit.run_keepfrac --configs <mnist cfg> <fmnist cfg> <cifar cfg> --out final_results\keepfrac --device cuda --dry-run
  (remove --dry-run to run; add --fractions 0.5 0.7 to leave out the 100% reference)
"""
import argparse
import json
import os
import sys

from . import run_battery as rb
from . import run_retest  # noqa: F401  (registers arms N, O, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join("final_results", "keepfrac"))
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    ap.add_argument("--arms", default="VNOPM")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion-mnist", "cifar10"])
    ap.add_argument("--contrastive-lambda", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base = {}
    for p in a.configs:
        with open(p) as f:
            c = json.load(f)
        if c["dataset"] in a.datasets:
            base[c["dataset"]] = c
    missing = [d for d in a.datasets if d not in base]
    if missing:
        sys.exit(f"no config given for: {missing}")
    for ds, c in base.items():
        if not c.get("minority_class"):
            sys.exit(f"config for {ds} has no minority_class set; use the config of your frog / shirt / digit 8 runs")

    for fr in a.fractions:
        tag = f"k{int(round(fr * 100))}"
        cfg_dir = os.path.join(a.out, "_configs", tag)
        os.makedirs(cfg_dir, exist_ok=True)
        paths = []
        for ds, c in base.items():
            c2 = dict(c)
            c2["minority_keep_fraction"] = fr
            p = os.path.join(cfg_dir, f"{ds}_config.json")
            with open(p, "w") as f:
                json.dump(c2, f, indent=2)
            paths.append(p)
        argv = ["run_battery", "--configs", *paths, "--out", os.path.join(a.out, tag), "--device", a.device,
                "--arms", a.arms, "--contrastive-lambda", str(a.contrastive_lambda),
                "--seeds", *[str(s) for s in a.seeds], "--split", "iid", "--datasets", *base.keys()]
        if a.dry_run:
            argv.append("--dry-run")
        print(f"=== keep fraction {fr} -> {tag} ===")
        sys.argv = argv
        rb.main()


if __name__ == "__main__":
    main()
