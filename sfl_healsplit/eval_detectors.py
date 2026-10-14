"""
Compare detectors and threshold rules on dumped smashed data (see dump_smashed.py).

    python -m sfl_healsplit.eval_detectors lab_dumps [more dump folders or .npz files] \
        [--sigmas 0.5,1,3] [--main-sigma 3] [--ceiling 0.5] [--csv lab_results.csv]

For every pool in every dump it builds
  - a CLEAN view (the data as the server received it, no attack), used for false positives and for how
    unevenly the flags fall across classes, and
  - ATTACK views, where each malicious client's unit offset is scaled by sigma * z_std0 and added to
    its samples (exactly what attacks.add_smashed_poison does), used for AUROC, precision and recall.
Offsets are re-created from the stored unit vectors, so any sigma can be tested from one dump.

Reported per dataset and split, pooled over seeds, checkpoints and pools:
  AUROC   how well the raw suspicion score separates poisoned from clean samples (no threshold involved)
  recall / precision / flagged   at the threshold rule, attack view
  clean flagged   share of samples flagged when there is NO attack (false positives)
  rare / maxcls   clean-view flag rate of the minority class, and of the most-flagged class
tas_as_is is the detector and rule exactly as run so far (detection.detect_poisoned).
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import rankdata

from .detectors import DETECTORS, RULES, as_is_flags


def auroc(score, gt):
    n1 = int(gt.sum())
    n0 = len(gt) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[gt].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def find_dumps(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "*.npz"))) if os.path.isdir(p) else [p]
    return out


def evaluate_dump(path, sigmas, ceiling, detectors):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    mal = list(meta["malicious_ids"])
    unit = {m: torch.tensor(d["unit_deltas"][i]) for i, m in enumerate(d["unit_ids"].tolist())}
    split = "iid" if meta["iid"] else "noniid"
    nc = meta["num_classes"]
    rows, classes = [], []
    for p in range(d["Z"].shape[0]):
        Z = torch.tensor(d["Z"][p])
        y = torch.tensor(d["y"][p]).long()
        cid = torch.tensor(d["cid"][p]).long()
        gt = np.isin(cid.numpy(), mal)
        views = {"clean": Z}
        for s in sigmas:
            Za = Z.clone()
            for m in mal:
                Za[cid == m] += s * meta["z_std0"] * unit[m]
            views[s] = Za
        base = dict(dataset=meta["dataset"], split=split, seed=meta["seed"], ckpt=int(d["round"][p]), pool=p)

        def add(det, rule, view, flagged, score):
            sig = view if view != "clean" else 0.0
            r = dict(base, detector=det, rule=rule, sigma=sig, flagged_frac=float(flagged.mean()))
            if view == "clean":
                r.update(auroc=float("nan"), precision=float("nan"), recall=float("nan"))
                yy = y.numpy()
                for c in range(nc):
                    m = yy == c
                    classes.append(dict(base, detector=det, rule=rule, cls=c,
                                        flagged=int(flagged[m].sum()), total=int(m.sum()),
                                        minority=int(c == meta["minority_class"])))
            else:
                tp = int((flagged & gt).sum())
                r.update(auroc=auroc(score, gt) if score is not None else float("nan"),
                         precision=tp / flagged.sum() if flagged.sum() else float("nan"),
                         recall=tp / gt.sum())
            rows.append(r)

        for view, Zv in views.items():
            add("tas_as_is", "as_is", view, as_is_flags(Zv), None)
            for name in detectors:
                s = DETECTORS[name](Zv, y, cid)
                for rname, rule in RULES.items():
                    add(name, rname, view, rule(s, ceiling=ceiling), s)
    return rows, classes


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def fmt(x, pct=True):
    return "   -  " if x != x else (f"{100 * x:5.1f}%" if pct else f"{x:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--sigmas", default="0.5,1,3")
    ap.add_argument("--main-sigma", type=float, default=3.0)
    ap.add_argument("--ceiling", type=float, default=0.5)
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--csv", default="lab_results.csv")
    a = ap.parse_args()
    sigmas = [float(s) for s in a.sigmas.split(",")]
    dets = [x for x in a.detectors.split(",") if x]
    files = find_dumps(a.paths)
    print(f"{len(files)} dump file(s)")
    rows, classes = [], []
    for f in files:
        r, c = evaluate_dump(f, sigmas, a.ceiling, dets)
        rows += r
        classes += c
        print(f"  evaluated {os.path.basename(f)}")
    keys = list(rows[0].keys())
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.csv}")

    groups = sorted({(r["dataset"], r["split"]) for r in rows})
    for ds, sp in groups:
        print(f"\n=== {ds} / {sp}  (attack sigma {a.main_sigma}; pooled over seeds, checkpoints, pools) ===")
        print(f"{'detector':16s} {'rule':8s} {'AUROC':>7s} {'recall':>7s} {'prec':>7s} {'flagged':>8s} | "
              f"{'clean flagged':>13s} {'rare cls':>8s} {'max cls':>8s}")
        combos = sorted({(r["detector"], r["rule"]) for r in rows if (r["dataset"], r["split"]) == (ds, sp)},
                        key=lambda t: (t[0] != "tas_as_is", t[0], t[1]))
        for det, rule in combos:
            sel = [r for r in rows if (r["dataset"], r["split"], r["detector"], r["rule"]) == (ds, sp, det, rule)]
            att = [r for r in sel if r["sigma"] == a.main_sigma]
            cln = [r for r in sel if r["sigma"] == 0.0]
            cc = [c for c in classes if (c["dataset"], c["split"], c["detector"], c["rule"]) == (ds, sp, det, rule)]
            per = defaultdict(lambda: [0, 0])
            rare = [0, 0]
            for c in cc:
                per[c["cls"]][0] += c["flagged"]
                per[c["cls"]][1] += c["total"]
                if c["minority"]:
                    rare[0] += c["flagged"]
                    rare[1] += c["total"]
            maxc = max((v[0] / v[1] for v in per.values() if v[1]), default=float("nan"))
            print(f"{det:16s} {rule:8s} {fmt(mean([r['auroc'] for r in att]), False):>7s} "
                  f"{fmt(mean([r['recall'] for r in att])):>7s} {fmt(mean([r['precision'] for r in att])):>7s} "
                  f"{fmt(mean([r['flagged_frac'] for r in att])):>8s} | {fmt(mean([r['flagged_frac'] for r in cln])):>13s} "
                  f"{fmt(rare[0] / rare[1] if rare[1] else float('nan')):>8s} {fmt(maxc):>8s}")
        print(f"\n  AUROC by attack strength (score only, no threshold)")
        print(f"  {'detector':16s} " + " ".join(f"{'sigma ' + str(s):>9s}" for s in sigmas))
        for det in ["tas_current"] + [x for x in dets if x != "tas_current"]:
            vals = []
            for s in sigmas:
                sel = [r["auroc"] for r in rows if (r["dataset"], r["split"], r["detector"], r["rule"], r["sigma"])
                       == (ds, sp, det, "top20", s)]
                vals.append(mean(sel))
            print(f"  {det:16s} " + " ".join(f"{fmt(v, False):>9s}" for v in vals))
        for rname in ("mad3.5", "mad6", "valley"):
            print(f"\n  recall / precision at rule {rname}, by attack strength")
            print(f"  {'detector':16s} " + " ".join(f"{'sigma ' + str(s):>15s}" for s in sigmas))
            for det in [x for x in dets]:
                cells = []
                for s in sigmas:
                    sel = [r for r in rows if (r["dataset"], r["split"], r["detector"], r["rule"], r["sigma"])
                           == (ds, sp, det, rname, s)]
                    cells.append(f"{fmt(mean([r['recall'] for r in sel]))}/{fmt(mean([r['precision'] for r in sel]))}")
                print(f"  {det:16s} " + " ".join(f"{c:>15s}" for c in cells))


if __name__ == "__main__":
    main()
