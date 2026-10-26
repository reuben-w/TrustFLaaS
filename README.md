# TrustFLaaS

Code for the paper *TrustFLaaS: A Self-Healing Federated Learning as a Service Defence for Trustworthy Collaborative Intelligence* (split federated learning with client-shift detection, DALSE healing, a logit-adjusted objective and a counts-only transparency log).

TrustFLaaS is arm **C** in the code: the client-shift detector, DALSE healing with a label validator, the logit-adjusted (LA) loss and the counts-only log. Arm **E** is the same system run with no attacker. With no client flagged, TrustFLaaS trains exactly as the LA loss does.

## What is here

| Path | Contents |
|---|---|
| `TrustFLaaS/` | The method and the experiment runners (`main.py` for one run, `run_battery.py` for the paper's attack arms, `run_retest.py` and `run_keepfrac.py` for the no-attacker comparisons).    `dump_smashed.py` and `eval_detectors.py` are the offline detector comparison (AUROC) cited in the paper; its output is in `results_summary/detector_comparison/` (24 dumps: 4 datasets, IID and non-IID, seeds 42 to 44, sigma 3). |
| `src/` | Dataset loading and seeding helpers used by `main.py`. |
| `configs/` | One base config per dataset (minority class kept at 20%). |
| `scripts/` | Table, check and timing scripts (`summarize_results.py`, `retest_keepfrac.py`, `verify_run.py`, `time_stages.py`). |
| `results_summary/` | The summary CSVs behind the paper's tables. |

The raw per-run results (confusion matrices, configs, metrics for every run) are in `trustflaas_raw_results.zip`, attached to the release (link: https://github.com/reuben-w/TrustFLaaS/releases/tag/1.0).

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
All experiments in the paper ran on one NVIDIA RTX 3090 (`--device cuda`). MNIST, Fashion-MNIST and CIFAR-10 download automatically through torchvision. PAMAP2 must be downloaded from the UCI Machine Learning Repository and extracted so that `<data-root>/PAMAP2_Dataset/Protocol/subject10X.dat` exists.

## Reproducing the experiments

Run from the repository root (the folder that contains the `TrustFLaaS/` package and `src/`). Seeds 42 to 46, 10 rounds, 5 clients, one malicious client (Gaussian smashed-data poisoning, sigma = 3). The malicious client is chosen from the seed (clients 2, 3, 4, 2 and 0 for seeds 42 to 46).

```
:: attack arms and ablations (A to H, J, U, V; E is TrustFLaaS with no attacker), IID and non-IID
python -m TrustFLaaS.run_battery --configs configs\mnist_config.json configs\fashion-mnist_config.json configs\cifar10_config.json configs\pamap2_config.json --out final_results --device cuda --arms ABCDEFGHJUV --seeds 42 43 44 45 46 --split iid
:: class-imbalance baselines without an attacker (plain split learning with LA, CB, LA+EWC)
python -m TrustFLaaS.run_retest --configs configs\mnist_config.json configs\fashion-mnist_config.json configs\cifar10_config.json configs\pamap2_config.json --out final_results\retest --device cuda --arms NOP --seeds 42 43 44 45 46 --split iid
:: minority-retention sweep (50%, 70%, 100% kept)
python -m TrustFLaaS.run_keepfrac --configs configs\mnist_config.json configs\fashion-mnist_config.json configs\cifar10_config.json --out final_results\keepfrac --device cuda
```
Add `--split noniid` for the Dirichlet alpha = 0.5 runs. Arm D (drop flagged data instead of healing) runs for seed 42 only.

Arm letters:

| Arm | Method |
|---|---|
| A | Generator-based healing system (HealSplit) |
| B | Client-shift detector + generator healing |
| **C** | **TrustFLaaS**: client-shift detector + DALSE healing + LA loss |
| D | C, dropping flagged data instead of healing (one seed) |
| E | C with no attacker |
| F | BAGAN-GP healing |
| G | DALSE + class-balanced (CB) loss |
| H | DALSE + LA loss + EWC |
| J | DALSE without a class-aware loss |
| U / V | Undefended, attacked / clean |
| N / O / P | Plain split learning with LA / CB / LA + EWC (no attacker) |

Tables:
```
python scripts\retest_keepfrac.py --existing final_results --retest final_results\retest --keepfrac final_results\keepfrac --out final_results\keepfrac
python -m TrustFLaaS.retest_metrics --existing final_results --retest final_results\retest --out final_results\retest
python scripts\summarize_results.py final_results --csv final_results\main_battery_summary.csv
```
Check the exact flags with `--help`; the commands above are the ones used for the paper.

## Notes

- PAMAP2's minority class (rope jumping) is naturally rare and is not subsampled; `--minority-keep-fraction` has no effect on it.
- The GPU is not bit-reproducible, so a rerun of a seed can differ slightly from the reported numbers.

## Datasets

MNIST (LeCun et al., 1998), Fashion-MNIST (Xiao et al., 2017) and CIFAR-10 (Krizhevsky, 2009) are downloaded through torchvision and are not redistributed here. PAMAP2 (Reiss and Stricker, 2012) is available from the UCI Machine Learning Repository; download it yourself and follow its terms of use. Each dataset keeps its own licence.

## Acknowledgements

This code re-implements, from the papers, methods proposed by others; no code from their repositories is included.

- **HealSplit** (Xie and Lyu, AAAI 2026): the topology-aware detector and the generator-based healing system used as a comparison (arm A, and the generator in arm B).
- **DALSE** (Chang et al., 2026): the drift-aware latent state estimation that our healer adapts from per-client weight drift to per-class smashed-data drift.
- **BAGAN-GP** (after Huang and Jafari, "Enhanced Balancing GAN"; gradient-penalty critic after Gulrajani et al., NeurIPS 2017): the class-conditioned generator in arm F, implemented from its core mechanism and not as a full reproduction.
- **Generative adversarial networks** (Goodfellow et al., NeurIPS 2014): the standard non-saturating GAN objective used by the generator-based healer.
- **Logit adjustment** (Menon et al., ICLR 2021), **class-balanced loss** (Cui et al., CVPR 2019), **elastic weight consolidation** (Kirkpatrick et al., PNAS 2017) and its use against class imbalance in MConFL (Yuan et al., 2026): the class-imbalance baselines and components.
- **Federated and split learning**: FedAvg (McMahan et al., 2017), split learning (Vepakomma et al., 2018) and SplitFed (Thapa et al., AAAI 2022).

The smashed-data poisoning attack follows the existing literature; we introduce no new attack.

## Licence
MIT
