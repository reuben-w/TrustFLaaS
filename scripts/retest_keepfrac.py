r"""
Tables for the minority-subsampling sweep, plus the "minority share of the TEST set" analysis. Reads files only.
  python retest_keepfrac.py --existing final_results --retest final_results\retest --keepfrac final_results\keepfrac --out final_results\keepfrac

Part 1  (keep fraction): for 20% (existing runs: V, M in --existing; N, O, P in --retest) and 50/70/100% (--keepfrac\kNN):
   overall accuracy on the full test set, and minority-class-only accuracy (= minority recall), mean +/- std over seeds.
   -> keepfrac_summary.csv, keepfrac_table.tex
Part 2  (test prior): the same trained models re-scored as if the minority class made up q of the test set
   (q = 5, 20, 50, 70%; the other classes share the rest equally). Recall does not depend on q; precision and accuracy do.
   Computed from each run's saved confusion matrix, so no retraining. -> prior_shift_summary.csv, prior_shift_table.tex
IID runs only. Minority class indices: MNIST 8, Fashion-MNIST 6, CIFAR-10 6.
"""
import argparse, csv, glob, json, os, re
import numpy as np

MINORITY = {"mnist": 8, "fashion-mnist": 6, "cifar10": 6}
DS = [("mnist", "MNIST", "digit 8"), ("fashion-mnist", "F-MNIST", "shirt"), ("cifar10", "CIFAR-10", "frog")]
METHODS = [("V", "Plain SFL"), ("N", "LA loss"), ("O", "CB loss"), ("P", "LA loss + EWC")]
RUN_RE = re.compile(r"^(.+?)_([A-Z])_seed(\d+)$")          # IID names only (no _noniid, no tags)
FRACS = [20, 50, 70, 100]
QS = [0.05, 0.20, 0.50, 0.70, 1.00]


def load(root, frac, arms=None):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*", "*_seed*"))):
        run = os.path.basename(d)
        m = RUN_RE.match(run)
        p = os.path.join(d, run + "_confusion_matrix.json")
        if not (m and os.path.isfile(p)) or m.group(1) not in MINORITY:
            continue
        if arms and m.group(2) not in arms:
            continue
        out.append(dict(ds=m.group(1), arm=m.group(2), seed=int(m.group(3)), frac=frac,
                        conf=np.array(json.load(open(p))["matrix"], dtype=float)))
    return out


def scores(conf, mc, q=None):
    n = conf.shape[0]
    rows = conf.sum(axis=1)
    rec = np.diag(conf) / rows
    pm = conf[:, mc] / rows                          # P(predicted minority | true class c)
    others = [c for c in range(n) if c != mc]
    if q is None:                                    # natural test set
        acc = np.trace(conf) / conf.sum()
        prec = conf[mc, mc] / conf[:, mc].sum() if conf[:, mc].sum() else 0.0
    else:                                            # minority is q of the test set, others share 1-q equally
        w_o = (1 - q) / len(others)
        acc = q * rec[mc] + w_o * rec[others].sum()
        tp, fp = q * rec[mc], w_o * pm[others].sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
    return acc, rec[mc], prec


def ms(v):
    return 100 * np.mean(v), 100 * (np.std(v, ddof=1) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", default="final_results")
    ap.add_argument("--retest", default=os.path.join("final_results", "retest"))
    ap.add_argument("--keepfrac", default=os.path.join("final_results", "keepfrac"))
    ap.add_argument("--out", default=os.path.join("final_results", "keepfrac"))
    a = ap.parse_args()
    runs = load(a.existing, 20, arms="V") + load(a.retest, 20, arms="NOP")
    for f in FRACS[1:]:
        runs += load(os.path.join(a.keepfrac, f"k{f}"), f)
    os.makedirs(a.out, exist_ok=True)

    def sel(ds, arm, f):
        return [r for r in runs if r["ds"] == ds and r["arm"] == arm and r["frac"] == f]

    # ---- part 1
    rows, cell = [], {}
    for ds, _, _ in DS:
        for arm, _ in METHODS:
            for f in FRACS:
                g = sel(ds, arm, f)
                if not g:
                    continue
                s = [scores(r["conf"], MINORITY[ds]) for r in g]
                acc, rec = ms([x[0] for x in s]), ms([x[1] for x in s])
                prec = ms([x[2] for x in s])
                cell[(ds, arm, f)] = (acc, rec)
                rows.append([ds, arm, f, len(g), *[round(v, 3) for v in (*acc, *rec, *prec)]])
    with open(os.path.join(a.out, "keepfrac_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ds", "arm", "keep_pct", "n_seeds", "acc_mean", "acc_std", "minority_recall_mean", "minority_recall_std",
                    "minority_precision_mean", "minority_precision_std"])
        w.writerows(rows)
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Varying the minority-class share kept in \\emph{training} (20\\%, 50\\%, 70\\%, 100\\% of its original size; 100\\% is the unsubsampled reference). IID partition, no attacker, mean $\\pm$ sample standard deviation over five seeds. "
         "Each trained model is tested twice. All: accuracy on the full, unsubsampled test set (all classes). Min.: accuracy on a test set containing only the minority class (its recall). LA: logit-adjusted. CB: class-balanced. Bold: highest minority-class mean among the three methods other than plain SFL; many differences are within one standard deviation. 100\\% kept is the unsubsampled reference. With no client flagged, \\sys reduces to the LA loss row.}",
         "\\label{tab:keepfrac}", "\\begin{tabular}{l" + "rr" * 4 + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{f}\\% kept in training}}" for f in FRACS) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(4)),
         "Method & " + " & ".join(["All", "Min."] * 4) + " \\\\", "\\midrule"]
    for k, (ds, name, cls) in enumerate(DS):
        L.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{name} (minority class: {cls})}}}} \\\\")
        best = {}
        for f in FRACS:
            for j in (0, 1):
                c = [cell[(ds, a_, f)][j][0] for a_, _ in METHODS[1:] if (ds, a_, f) in cell]
                best[(f, j)] = max(c) if c else None
        for arm, nm in METHODS:
            parts = []
            for f in FRACS:
                for j in (0, 1):
                    if (ds, arm, f) not in cell:
                        parts.append("--"); continue
                    m_, s_ = cell[(ds, arm, f)][j]
                    t = f"{m_:.1f}$\\pm${s_:.1f}"
                    if j == 1 and arm != "V" and best[(f, j)] is not None and abs(m_ - best[(f, j)]) < 1e-9:
                        t = f"\\textbf{{{m_:.1f}}}$\\pm${s_:.1f}"
                    parts.append(t)
            L.append(f"{nm} & " + " & ".join(parts) + " \\\\")
        if k < len(DS) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(os.path.join(a.out, "keepfrac_table.tex"), "w").write("\n".join(L) + "\n")

    # ---- part 2: minority share of the test set, models trained at 20% kept
    rows, cell2 = [], {}
    for ds, _, _ in DS:
        for arm, _ in METHODS:
            g = sel(ds, arm, 20)
            if not g:
                continue
            for q in QS:
                s = [scores(r["conf"], MINORITY[ds], q) for r in g]
                acc, prec = ms([x[0] for x in s]), ms([x[2] for x in s])
                cell2[(ds, arm, q)] = (acc, prec)
                rows.append([ds, arm, q, len(g), *[round(v, 3) for v in (*acc, *prec)]])
    with open(os.path.join(a.out, "prior_shift_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ds", "arm", "minority_share_of_test", "n_seeds", "acc_mean", "acc_std", "minority_precision_mean", "minority_precision_std"])
        w.writerows(rows)
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Varying the minority-class share of the \\emph{test} set. All models were trained with 20\\% of the minority class kept (IID, no attacker) and are scored as if the minority class made up 5\\%, 20\\%, 50\\%, 70\\% or 100\\% of the test set, the other classes sharing the rest equally. At 100\\% the test set is the minority class alone, so accuracy equals minority recall and precision is undefined (--). "
         "Minority recall does not change with this share. Acc.: overall accuracy. Prec.: minority-class precision. Mean $\\pm$ sample standard deviation over five seeds. Computed from each run's confusion matrix. With no client flagged, \\sys reduces to the LA loss row.}",
         "\\label{tab:priorshift}", "\\begin{tabular}{l" + "rr" * 5 + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{int(q*100)}\\% minority in test}}" for q in QS) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(5)),
         "Method & " + " & ".join(["Acc.", "Prec."] * 5) + " \\\\", "\\midrule"]
    for k, (ds, name, cls) in enumerate(DS):
        L.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{name} (minority class: {cls})}}}} \\\\")
        for arm, nm in METHODS:
            parts = []
            for q in QS:
                if (ds, arm, q) not in cell2:
                    parts += ["--", "--"]; continue
                (am, asd), (pm_, psd) = cell2[(ds, arm, q)]
                parts += [f"{am:.1f}$\\pm${asd:.1f}", ("--" if q >= 1 else f"{pm_:.1f}$\\pm${psd:.1f}")]
            L.append(f"{nm} & " + " & ".join(parts) + " \\\\")
        if k < len(DS) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    open(os.path.join(a.out, "prior_shift_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"{len(runs)} runs read")
    for ds, _, _ in DS:
        for f in FRACS:
            n = {arm: len(sel(ds, arm, f)) for arm, _ in METHODS}
            if any(v < 5 for v in n.values()):
                print(f"  {ds} {f}% kept: seeds per method {n}")


if __name__ == "__main__":
    main()
