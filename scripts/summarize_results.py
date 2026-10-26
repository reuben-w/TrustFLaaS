"""
Summarise a final_results folder (made by run_battery.py) into paper-ready tables. Reads files only.
  python summarize_results.py final_results [--nodef mnist_nodefense_results fmnist_nodefense_results ...] [--csv out.csv]
--nodef adds your undefended runs as arms U (attacked) and V (clean), seeds 42-44 only, so n matches.
Tables: final accuracy, minority-class accuracy, detection quality, healing outcome. Mean +/- std over seeds.
"""
import argparse, csv, glob, json, os, re
import numpy as np

MINORITY = {"mnist": 8, "fashion-mnist": 6, "cifar10": 6, "pamap2": 11}   # index of the minority class
NAMES = {"U": "U undefended, attacked", "V": "V undefended, clean",
         "A": "A HealSplit (TAS+GAN)", "B": "B shift-check+GAN", "C": "C TrustFLaaS", "D": "D TrustFLaaS, drop",
         "E": "E TrustFLaaS, no attack", "F": "F BAGAN-GP healing", "G": "G DALSE+CB loss",
         "H": "H DALSE+LA+EWC", "J": "J DALSE, no class loss"}
ARM_ORDER = "UVABCDEFGHJ"


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse(run):                      # mnist_C_noniid_seed42 -> (mnist, C, noniid, 42)
    m = re.match(r"(.+?)_([A-Z])(_noniid)?(_s[\d.]+)?_seed(\d+)$", run)
    return m.group(1), m.group(2), ("noniid" if m.group(3) else "iid") + (m.group(4) or ""), int(m.group(5))


def load(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*", "*_seed*"))):
        if not os.path.isdir(d):
            continue
        run = os.path.basename(d)
        ds, arm, split, seed = parse(run)
        cfg = json.load(open(os.path.join(d, run + "_config.json")))
        met = read_rows(os.path.join(d, run + "_metrics.csv"))
        pc = read_rows(os.path.join(d, run + "_per_class_accuracy.csv"))
        last, last3 = met[-1], met[-3:]
        mc = MINORITY[ds]
        r = dict(run=run, ds=ds, arm=arm, split=split, seed=seed, rounds=int(last["round"]),
                 acc=float(last["test_accuracy"]),
                 acc3=np.mean([float(x["test_accuracy"]) for x in last3]),
                 minority=float(pc[-1][f"class{mc}_acc"]),
                 minority3=np.mean([float(x[f"class{mc}_acc"]) for x in pc[-3:]]))
        mal = set(cfg.get("malicious_client_ids", []))
        rr = [x for x in met if x["round"] != "0"]
        fl = sum(int(x["num_flagged"]) for x in rr)
        r["flag_frac"] = fl / max(sum(int(x["num_total"]) for x in rr), 1)
        r["healed_frac"] = sum(int(x["num_replaced"]) for x in rr) / max(fl, 1) if fl else np.nan
        r["dropped_frac"] = sum(int(x["num_dropped"]) for x in rr) / max(fl, 1) if fl else np.nan
        cdp = os.path.join(d, run + "_metrics_client_detection.csv")
        if os.path.exists(cdp):
            ok = false_flag = missed = n = 0
            for x in read_rows(cdp):
                got = {int(c) for c in x["flagged_clients"].split()} if x["flagged_clients"].strip() else set()
                n += 1
                ok += (got == mal) if mal else (not got)
                false_flag += bool(got - mal)
                missed += bool(mal - got)
            r.update(det_rounds=n, det_correct=ok, det_false=false_flag, det_missed=missed)
        elif arm == "A":                                    # TAS: share of the malicious client's pool flagged
            tot = hit = other_tot = other_hit = 0
            for rd in range(1, r["rounds"] + 1):
                for c in range(cfg["num_clients"]):
                    p = os.path.join(d, "transparency_reports", f"client_{c}", f"round_{rd:03d}.json")
                    if not os.path.exists(p):
                        continue
                    n_fl = sum(v["flagged_this_round"] for v in json.load(open(p))["report"].values())
                    if c in mal:
                        tot += 128; hit += n_fl
                    else:
                        other_tot += 128; other_hit += n_fl
            r["tas_mal"] = hit / tot if tot else np.nan
            r["tas_other"] = other_hit / other_tot if other_tot else np.nan
        runs.append(r)
    return runs


def ms(vals, pct=True):
    v = np.array([x for x in vals if x == x], dtype=float)
    if not len(v):
        return "-"
    s = 100 if pct else 1
    return f"{v.mean() * s:.1f}" + (f" ± {v.std() * s:.1f}" if len(v) > 1 else "") + f" (n={len(v)})"


def table(runs, title, getter, pct=True):
    print(f"\n=== {title} ===")
    for split in sorted({r["split"] for r in runs}):
        sub = [r for r in runs if r["split"] == split]
        if not sub:
            continue
        dss = sorted({r["ds"] for r in sub})
        print(f"[{split}]  " + " | ".join(f"{d:>22}" for d in dss))
        for arm in ARM_ORDER:
            cells = []
            for d in dss:
                cells.append(ms([getter(r) for r in sub if r["ds"] == d and r["arm"] == arm and getter(r) is not None], pct))
            if any(c != "-" for c in cells):
                print(f"{NAMES[arm]:<24}" + " | ".join(f"{c:>22}" for c in cells))


def load_nodef(dirs, seeds=(42, 43, 44)):
    """Undefended runs from battery_nodefense_baseline.bat: <ds>_..._nodef_{clean|sp}[_noniid]_seed<N>."""
    key = {"mnist": "mnist", "fmnist": "fashion-mnist", "cifar10": "cifar10", "pamap2": "pamap2"}
    runs = []
    for root in dirs:
        for d in sorted(glob.glob(os.path.join(root, "*_seed*"))):
            run = os.path.basename(d)
            m = re.match(r"([a-z0-9]+)_.*_nodef_(clean|sp)(_noniid)?_seed(\d+)$", run)
            if not m or int(m.group(4)) not in seeds or m.group(1) not in key:
                continue
            ds = key[m.group(1)]
            met = read_rows(os.path.join(d, run + "_metrics.csv"))
            pc = read_rows(os.path.join(d, run + "_per_class_accuracy.csv"))
            mc = MINORITY[ds]
            runs.append(dict(run=run, ds=ds, arm="U" if m.group(2) == "sp" else "V",
                             split="noniid" if m.group(3) else "iid", seed=int(m.group(4)),
                             rounds=int(met[-1]["round"]), acc=float(met[-1]["test_accuracy"]),
                             acc3=np.mean([float(x["test_accuracy"]) for x in met[-3:]]),
                             minority=float(pc[-1][f"class{mc}_acc"]),
                             minority3=np.mean([float(x[f"class{mc}_acc"]) for x in pc[-3:]]),
                             flag_frac=np.nan, healed_frac=np.nan, dropped_frac=np.nan))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--nodef", nargs="*", default=[], help="undefended result folders")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    runs = load(a.root) + load_nodef(a.nodef)
    print(f"{len(runs)} runs read from {a.root}")
    table(runs, "Final-round test accuracy (%)", lambda r: r["acc"])
    table(runs, "Final-round MINORITY-class accuracy (%)", lambda r: r["minority"])
    table(runs, "Healing outcome: share of flagged samples successfully replaced (%)", lambda r: r["healed_frac"])
    print("\n=== Detection (client-level arms B/C/D/E): rounds in which the flagged set was exactly right ===")
    for split in sorted({r["split"] for r in runs}):
        for d in sorted({r["ds"] for r in runs}):
            for arm in "BCDEFGHIJ":
                sub = [r for r in runs if r["ds"] == d and r["arm"] == arm and r["split"] == split and "det_rounds" in r]
                if sub:
                    n = sum(r["det_rounds"] for r in sub)
                    print(f"{split:6} {d:14} {arm}: exact {sum(r['det_correct'] for r in sub)}/{n}  "
                          f"wrongly-flagged {sum(r['det_false'] for r in sub)}  missed {sum(r['det_missed'] for r in sub)}")
    print("\n=== TAS (arm A): share of pool flagged, malicious client vs the others (chance = ~20% for both) ===")
    for split in sorted({r["split"] for r in runs}):
        for d in sorted({r["ds"] for r in runs}):
            sub = [r for r in runs if r["ds"] == d and r["arm"] == "A" and r["split"] == split and "tas_mal" in r]
            if sub:
                print(f"{split:6} {d:14} malicious {100*np.mean([r['tas_mal'] for r in sub]):.1f}%   others {100*np.mean([r['tas_other'] for r in sub]):.1f}%")
    if a.csv:
        keys = ["run", "ds", "arm", "split", "seed", "rounds", "acc", "acc3", "minority", "minority3", "flag_frac",
                "healed_frac", "dropped_frac", "det_rounds", "det_correct", "det_false", "det_missed", "tas_mal", "tas_other"]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(runs)
        print(f"\nper-run rows written to {a.csv}")


if __name__ == "__main__":
    main()
