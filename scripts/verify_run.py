"""
Verify freshly produced run folders (use on the smoke tests, then on batteries).

    python verify_run.py ./smoke_results
    python verify_run.py ./mnist_attack_results

For every run folder directly under the given directory it checks:
  1. <run>_config.json, <run>_metrics.csv, <run>_per_class_accuracy.csv,
     <run>_confusion_matrix.json and transparency_reports/{aggregate,client_*}
     all exist (same layout as the earlier batteries, plus the two new files)
  2. metrics.csv has num_rounds+1 data rows, no NaN loss/accuracy
  3. per-class file overall_acc == metrics.csv test_accuracy in EVERY round
     (proves the new evaluator reproduces engine.evaluate exactly)
  4. per-class totals are identical across rounds (fixed test set) and the
     confusion matrix agrees with the last per-class row
  5. attack runs: config says attack=True / smashed_poison with >=1 malicious
     client; no-attack runs: attack=False and no malicious clients
Exit code 1 if anything fails.
"""
import csv
import glob
import json
import math
import os
import sys


def check_run(run_dir):
    problems = []
    run = os.path.basename(os.path.normpath(run_dir))
    cfg_p = os.path.join(run_dir, f"{run}_config.json")
    met_p = os.path.join(run_dir, f"{run}_metrics.csv")
    pc_p = os.path.join(run_dir, f"{run}_per_class_accuracy.csv")
    cm_p = os.path.join(run_dir, f"{run}_confusion_matrix.json")
    for p in (cfg_p, met_p, pc_p, cm_p):
        if not os.path.exists(p):
            problems.append(f"missing {os.path.basename(p)}")
    tr = os.path.join(run_dir, "transparency_reports")
    try:
        defended = bool(json.load(open(cfg_p)).get("defense", True))
    except Exception:
        defended = True   # unreadable config -> be strict
    if defended:   # undefended (--no-defense) runs write no transparency reports
        if not os.path.isdir(os.path.join(tr, "aggregate")):
            problems.append("missing transparency_reports/aggregate")
        if not glob.glob(os.path.join(tr, "client_*")):
            problems.append("missing transparency_reports/client_*")
    if problems:
        return problems

    cfg = json.load(open(cfg_p))
    metrics = list(csv.DictReader(open(met_p)))
    per_class = list(csv.DictReader(open(pc_p)))
    num_rounds = cfg["num_rounds"]

    if len(metrics) != num_rounds + 1:
        problems.append(f"metrics rows={len(metrics)} expected {num_rounds + 1}")
    if len(per_class) != len(metrics):
        problems.append(f"per_class rows={len(per_class)} != metrics rows={len(metrics)}")
    for m in metrics:
        for k in ("test_loss", "test_accuracy"):
            if not math.isfinite(float(m[k])):
                problems.append(f"non-finite {k} in round {m['round']}")

    for m, pcrow in zip(metrics, per_class):
        if pcrow["overall_acc"] == "":
            problems.append(f"empty overall_acc round {pcrow['round']}")
        elif abs(float(m["test_accuracy"]) - float(pcrow["overall_acc"])) > 1e-6:
            problems.append(
                f"round {m['round']}: metrics acc {m['test_accuracy']} != "
                f"per-class overall {pcrow['overall_acc']}")

    if per_class:
        ncls = sum(1 for k in per_class[0] if k.endswith("_total"))
        totals0 = [per_class[0][f"class{i}_total"] for i in range(ncls)]
        for r in per_class:
            if [r[f"class{i}_total"] for i in range(ncls)] != totals0:
                problems.append(f"per-class totals changed at round {r['round']}")
                break
        cm = json.load(open(cm_p))["matrix"]
        last = per_class[-1]
        for i in range(ncls):
            if sum(cm[i]) != int(last[f"class{i}_total"]) or cm[i][i] != int(last[f"class{i}_correct"]):
                problems.append(f"confusion matrix disagrees with per-class row for class {i}")
                break

    mal = cfg.get("malicious_client_ids", [])
    if cfg.get("attack"):
        if cfg.get("attack_type") != "smashed_poison":
            problems.append(f"attack_type={cfg.get('attack_type')}")
        if not mal:
            problems.append("attack=True but no malicious clients")
    elif mal:
        problems.append("attack=False but malicious clients listed")
    return problems


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "*_config.json")))
    if not runs:
        print(f"No run folders found under {root}")
        sys.exit(1)
    bad = 0
    for r in runs:
        probs = check_run(r)
        name = os.path.basename(r)
        if probs:
            bad += 1
            print(f"FAIL  {name}")
            for p in probs:
                print(f"        - {p}")
        else:
            cfg = json.load(open(os.path.join(r, f"{name}_config.json")))
            print(f"PASS  {name}  (attack={cfg['attack']}, malicious={cfg.get('malicious_client_ids')}, "
                  f"healing={cfg['healing_mode']}, ewc={cfg['ewc_lambda']}, cb={cfg['cb_loss_beta']}, "
                  f"la={cfg['la_loss_psi']}, contrastive={cfg['contrastive_lambda']}, iid={cfg['iid']})")
    print(f"\n{len(runs) - bad}/{len(runs)} runs passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
