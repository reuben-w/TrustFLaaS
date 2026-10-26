r"""
Stage-timing wrapper. NEW file: changes nothing else in the code base.

Runs ONE normal training run (same code path as the battery) and records, per round, how long each stage
of the server takes. Nothing about training is altered: it only wraps four functions with a stopwatch
(torch.cuda.synchronize before and after, so GPU work is counted).

Per round it records (seconds):
  round_total      train_one_round_with_healing as a whole (detection, healing, training, FedAvg,
                   log); the per-round evaluation on the test set is NOT included
  detect           client_shift_scores + flag_clients
  validator        validator training (stats["validator_train_time"])
  substitution     healing / substitution (stats["substitution_time"]; for the GAN, generator training is timed separately as gen_train)
  gen_train        generator training (stats["gen_train_time"]; GAN / BAGAN-GP only)
  other            round_total minus the five above (local training, FedAvg, bookkeeping)
  num_flagged      flagged samples in the round (the divisor for the per-flagged-sample cost)

Run from the folder that holds TrustFLaaS\ and src\ , with NOTHING else running on the GPU:

  python time_stages.py run --config <dataset *_config.json> --arm C --seed 42 --out timing --device cuda
  python time_stages.py summarize --out timing

Arms: C (final TrustFLaaS), A (GAN-based healing system), U (undefended; only whole-run time is recorded).
Use the same four dataset configs as the battery (the ones passed to run_battery --configs).
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time

KEYS = ["round_total", "detect", "validator", "substitution", "gen_train", "other", "num_flagged"]


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, bucket, key):
    def wrapper(*a, **k):
        sync(); t0 = time.perf_counter()
        out = fn(*a, **k)
        sync(); bucket[key] = bucket.get(key, 0.0) + time.perf_counter() - t0
        return out
    return wrapper


def do_run(args):
    # reuse the battery's arm definitions so the settings are identical to the reported runs
    from TrustFLaaS import run_battery as rb
    with open(args.config) as f:
        cfg = json.load(f)
    ds = cfg["dataset"]
    cfg.update(rb.COMMON)
    cfg.update(rb.ARMS[args.arm])
    cfg.update(seed=args.seed, device=args.device, results_dir=os.path.join(args.out, "runs", ds))
    cfg["iid"] = (args.split == "iid")
    if args.num_rounds:
        cfg["num_rounds"] = args.num_rounds
    cfg["la_counts"] = "full"
    cfg["run_name"] = f"{ds}_{args.arm}{'' if cfg['iid'] else '_noniid'}_timing_seed{args.seed}"
    os.makedirs(cfg["results_dir"], exist_ok=True)

    rounds, cur = [], {}
    from TrustFLaaS import defense, main as mainmod
    defense.client_shift_scores = timed(defense.client_shift_scores, cur, "detect")
    defense.flag_clients = timed(defense.flag_clients, cur, "detect")
    orig_round = defense.train_one_round_with_healing

    def round_wrapper(*a, **k):
        cur.clear()
        sync(); t0 = time.perf_counter()
        out = orig_round(*a, **k)
        sync(); total = time.perf_counter() - t0
        stats = out[1]
        row = dict(round_total=total, detect=cur.get("detect", 0.0),
                   validator=float(stats.get("validator_train_time", 0.0)),
                   substitution=float(stats.get("substitution_time", 0.0)),
                   gen_train=float(stats.get("gen_train_time", 0.0)),
                   num_flagged=int(stats.get("num_flagged", 0)))
        row["other"] = total - row["detect"] - row["validator"] - row["substitution"] - row["gen_train"]
        rounds.append(row)
        return out
    defense.train_one_round_with_healing = round_wrapper

    sys.argv = ["main"] + rb.argv_from(cfg)
    sync(); t_run = time.perf_counter()
    mainmod.main()
    sync(); run_total = time.perf_counter() - t_run

    path = os.path.join(args.out, f"{ds}_{args.arm}{'_noniid' if not cfg['iid'] else ''}_seed{args.seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arm", "seed", "round"] + KEYS + ["run_total"])
        for i, r in enumerate(rounds, 1):
            w.writerow([ds, args.arm, args.seed, i] + [f"{r[k]:.6f}" if k != "num_flagged" else r[k] for k in KEYS] + [f"{run_total:.3f}"])
        if not rounds:        # undefended: no per-round stages, keep the whole-run time
            w.writerow([ds, args.arm, args.seed, 0] + [""] * len(KEYS) + [f"{run_total:.3f}"])
    print(f"timing written to {path} (whole run {run_total:.1f} s)")


def do_summarize(args):
    rows = {}
    for p in sorted(glob.glob(os.path.join(args.out, "*.csv"))):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                rows.setdefault((r["dataset"], r["arm"]), []).append(r)
    print("Mean seconds per round, skipping round 1 (start-up cost). Per-sample = (validator + substitution + gen_train) / flagged.")
    print(f"{'dataset':14s}{'arm':>4s}{'round':>9s}{'detect':>9s}{'valid.':>9s}{'subst.':>9s}{'gen':>9s}{'other':>9s}{'flagged':>9s}{'ms/flagged':>12s}")
    for (ds, arm), rs in sorted(rows.items()):
        rs = [r for r in rs if r["round"] not in ("0", "1") and r["round_total"] != ""]
        if not rs:
            print(f"{ds:14s}{arm:>4s}   (whole run only: {rows[(ds, arm)][0]['run_total']} s)")
            continue
        m = {k: statistics.mean(float(r[k]) for r in rs) for k in KEYS}
        heal = m["validator"] + m["substitution"] + m["gen_train"]
        per = 1000 * heal / m["num_flagged"] if m["num_flagged"] > 0 else float("nan")
        print(f"{ds:14s}{arm:>4s}{m['round_total']:9.3f}{m['detect']:9.3f}{m['validator']:9.3f}{m['substitution']:9.3f}"
              f"{m['gen_train']:9.3f}{m['other']:9.3f}{m['num_flagged']:9.0f}{per:12.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--arm", default="C", choices=["C", "A", "U"])
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", default="timing")
    r.add_argument("--device", default="cuda")
    r.add_argument("--split", default="iid", choices=["iid", "noniid"])
    r.add_argument("--num-rounds", type=int, default=None)
    s = sub.add_parser("summarize")
    s.add_argument("--out", default="timing")
    a = ap.parse_args()
    do_run(a) if a.cmd == "run" else do_summarize(a)
