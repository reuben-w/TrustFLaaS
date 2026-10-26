r"""
Precision / recall tables for the retest. Reads files only; no training.
  python retest_metrics.py --existing final_results --retest final_results\retest --out final_results\retest

For every finished run it reads <run>_confusion_matrix.json (final round, rows = true class, cols = predicted) and computes
  accuracy; minority-class recall and precision; majority recall and precision (macro mean over all other classes).
Recall of class c = TP / (samples of c). Precision of c = TP / (samples predicted as c); 0 when nothing is predicted as c.
Then mean +/- sample std over seeds, and two LaTeX tables:
  retest_table_noattack.tex   TrustFLaaS (E) vs LA (N), CB (O), LA+EWC (P); no-defence clean (V) as a reference row
  retest_table_attack.tex     TrustFLaaS (C) vs GAN-based healing (A), BAGAN-GP (F), DALSE only (J); undefended attacked (U) as reference
Also retest_runs.csv (one row per run) and retest_summary.csv (one row per dataset / split / arm).
"""
import argparse, csv, glob, json, os, re
import numpy as np

MINORITY = {"mnist": 8, "fashion-mnist": 6, "cifar10": 6, "pamap2": 11}
DS = [("mnist", "MNIST", "digit 8"), ("fashion-mnist", "F-MNIST", "shirt"),
      ("pamap2", "PAMAP2", "rope jumping"), ("cifar10", "CIFAR-10", "frog")]
NO_ATTACK = [("V", "Plain SFL, no class loss", True), ("N", "LA loss", False), ("O", "CB loss", False),
             ("P", "LA loss + EWC", False), ("E", "\\textbf{TrustFLaaS}", False)]
ATTACK = [("U", "Undefended, attacked", True), ("A", "GAN-based healing system", False),
          ("F", "BAGAN-GP healing", False), ("J", "DALSE only", False), ("C", "\\textbf{TrustFLaaS}", False)]
KEYS = ["acc", "min_rec", "min_prec", "maj_rec", "maj_prec"]
RUN_RE = re.compile(r"^(.+?)_([A-Z])(_noniid)?_seed(\d+)$")      # tagged variants (_s3, _lakept) do not match


def metrics(conf, minority):
    m = np.array(conf, dtype=float)
    n = m.shape[0]
    tp = np.diag(m)
    rec = np.divide(tp, m.sum(axis=1), out=np.zeros(n), where=m.sum(axis=1) > 0)
    prec = np.divide(tp, m.sum(axis=0), out=np.zeros(n), where=m.sum(axis=0) > 0)
    maj = [i for i in range(n) if i != minority]
    return dict(acc=tp.sum() / m.sum(), min_rec=rec[minority], min_prec=prec[minority],
                maj_rec=rec[maj].mean(), maj_prec=prec[maj].mean())


def load(root, skip_sub=None):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "*_seed*"))):
        run = os.path.basename(d)
        mt = RUN_RE.match(run)
        p = os.path.join(d, run + "_confusion_matrix.json")
        if not (mt and os.path.isfile(p)):
            continue
        ds, arm, nonid, seed = mt.group(1), mt.group(2), mt.group(3), int(mt.group(4))
        if ds not in MINORITY:
            continue
        conf = json.load(open(p))["matrix"]
        r = dict(run=run, ds=ds, arm=arm, split="noniid" if nonid else "iid", seed=seed)
        r.update(metrics(conf, MINORITY[ds]))
        out.append(r)
    return out


def agg(rows, ds, arm, split):
    v = [r for r in rows if r["ds"] == ds and r["arm"] == arm and r["split"] == split]
    if not v:
        return None
    return {k: (100 * np.mean([r[k] for r in v]), 100 * (np.std([r[k] for r in v], ddof=1) if len(v) > 1 else 0.0))
            for k in KEYS} | {"n": len(v)}


def table(rows, spec, label, caption):
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\begin{tabular}{l" + "r" * 5 + "r" * 5 + "}", "\\toprule",
         " & \\multicolumn{5}{c}{IID partition} & \\multicolumn{5}{c}{Non-IID partition} \\\\",
         "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}",
         " & & \\multicolumn{2}{c}{Minority class} & \\multicolumn{2}{c}{Majority classes} & & \\multicolumn{2}{c}{Minority class} & \\multicolumn{2}{c}{Majority classes} \\\\",
         "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11}",
         "Method & Acc. & Rec. & Prec. & Rec. & Prec. & Acc. & Rec. & Prec. & Rec. & Prec. \\\\", "\\midrule"]
    for i, (ds, name, cls) in enumerate(DS):
        L.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{name} (minority class: {cls})}}}} \\\\")
        cells = {a: [agg(rows, ds, a, s) for s in ("iid", "noniid")] for a, _, _ in spec}
        best = {}
        for si in (0, 1):
            for k in KEYS:
                c = [cells[a][si][k][0] for a, _, ref in spec if not ref and cells[a][si]]
                best[(si, k)] = max(c) if c else None
        for a, nm, ref in spec:
            parts = []
            for si in (0, 1):
                s = cells[a][si]
                for k in KEYS:
                    if s is None:
                        parts.append("--"); continue
                    m, sd = s[k]
                    t = f"{m:.1f}$\\pm${sd:.1f}"
                    if not ref and best[(si, k)] is not None and abs(m - best[(si, k)]) < 1e-9:
                        t = f"\\textbf{{{m:.1f}}}$\\pm${sd:.1f}"
                    parts.append(t)
            L.append(f"{nm} & " + " & ".join(parts) + " \\\\")
        if i < len(DS) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default="final_results")
    ap.add_argument("--retest", default=os.path.join("final_results", "retest"))
    ap.add_argument("--out", default=os.path.join("final_results", "retest"))
    a = ap.parse_args()
    rows = load(a.existing) + load(a.retest)
    seen, uniq = set(), []
    for r in rows:                                   # a run found in both places is counted once
        if r["run"] not in seen:
            seen.add(r["run"]); uniq.append(r)
    rows = uniq
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "retest_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "ds", "arm", "split", "seed"] + KEYS)
        w.writeheader(); [w.writerow({k: r[k] for k in w.fieldnames}) for r in rows]
    with open(os.path.join(a.out, "retest_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ds", "split", "arm", "n"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "std")])
        for ds, _, _ in DS:
            for sp in ("iid", "noniid"):
                for arm in "VNOPEUAFJC":
                    g = agg(rows, ds, arm, sp)
                    if g:
                        w.writerow([ds, sp, arm, g["n"]] + [f"{g[k][i]:.3f}" for k in KEYS for i in (0, 1)])
    common = ("Final-round accuracy (Acc.), recall (Rec.) and precision (Prec.), in \\%, mean $\\pm$ sample standard deviation over five seeds (42--46), "
              "10 rounds, 5 clients. Minority class: the class named for each dataset. Majority classes: macro mean over all other classes. "
              "LA: logit-adjusted. CB: class-balanced. Bold: best mean among the compared methods in that column (reference row excluded). ")
    open(os.path.join(a.out, "retest_table_noattack.tex"), "w").write(table(
        rows, NO_ATTACK, "tab:noattack",
        common + "No attacker. Rows N--P use the same training code as TrustFLaaS with nothing flagged and no healing; "
                 "the first row has no class-aware loss. The TrustFLaaS row is the full system run with no attacker."))
    open(os.path.join(a.out, "retest_table_attack.tex"), "w").write(table(
        rows, ATTACK, "tab:attack",
        common + "One malicious client (smashed-data poisoning). Healing methods use the client-shift detector except the GAN-based healing system, "
                 "which uses its own detector."))
    print(f"{len(rows)} runs read; wrote retest_runs.csv, retest_summary.csv and two .tex tables to {a.out}")
    for sp in ("iid", "noniid"):
        for ds, _, _ in DS:
            miss = [x for x in "VNOPMUAFJI" if (agg(rows, ds, x, sp) or {"n": 0})["n"] < 5]
            if miss:
                print(f"  {ds} {sp}: fewer than 5 seeds for arms {''.join(miss)}")


if __name__ == "__main__":
    main()
