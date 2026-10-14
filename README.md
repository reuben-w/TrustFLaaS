# TrustFLaaS

Code for the paper *TrustFLaaS: Self-Healing FLaaS Against Poisoning Attacks for Trustworthy Federated Learning as a Service* (split federated learning with client-shift detection, DALSE healing, class-aware training and a counts-only transparency log).



## What is here

| Path | Contents |
|---|---|
| `sfl_healsplit/` | The method and the experiment runners (`main.py` for one run, `run_battery.py` for the paper's arms, `run_retest.py` and `run_keepfrac.py` for the extra experiments). |
| `configs/` | One base config per dataset (minority class kept at 20%). |
| `scripts/` | Table and timing scripts (`summarize_results.py`, `retest_keepfrac.py`, `time_stages.py`). |
| `analysis/` | Offline detector and smashed-data analyses. |
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

Run from the folder that contains `sfl_healsplit/`. Seeds 42 to 46, 10 rounds, 5 clients, client 2 malicious (Gaussian smashed-data poisoning, sigma = 3).

```
:: the paper's attack arms and ablations (A to M, U, V), IID and non-IID
python -m sfl_healsplit.run_battery --configs configs\mnist_config.json configs\fashion-mnist_config.json configs\cifar10_config.json configs\pamap2_config.json --out final_results --device cuda --arms ABCDEFGHIJKMUV --contrastive-lambda 1.0 --seeds 42 43 44 45 46 --split iid
:: class-imbalance baselines without an attacker (plain split learning with LA, CB, LA+EWC)
python -m sfl_healsplit.run_retest --configs configs\... --out final_results\retest --device cuda --arms NOP --seeds 42 43 44 45 46 --split iid
:: minority-retention sweep (50%, 70%, 100% kept)
python -m sfl_healsplit.run_keepfrac --configs configs\mnist_config.json configs\fashion-mnist_config.json configs\cifar10_config.json --out final_results\keepfrac --device cuda
```
Arm letters: A generator-based system (HealSplit), B client-shift detector + generator, C DALSE + LA loss, D drop flagged data, E C without attacker, F BAGAN-GP, G DALSE + CB loss, H DALSE + LA + EWC, I TrustFLaaS (C + contrastive step), J DALSE without a class loss, K kept counts, M I without attacker, U undefended under attack, V undefended clean; N, O, P are plain split learning with LA, CB and LA+EWC (no attacker). Add `--split noniid` for the Dirichlet alpha = 0.5 runs.

Tables:
```
python scripts\retest_keepfrac.py --existing final_results --retest final_results\retest --keepfrac final_results\keepfrac --out final_results\keepfrac
python -m sfl_healsplit.retest_metrics --existing final_results --retest final_results\retest --out final_results\retest
```
Check the exact flags with `--help`; the commands above are the ones used for the paper.

## Notes

- PAMAP2's minority class (rope jumping) is naturally rare and is not subsampled; `--minority-keep-fraction` has no effect on it.
- `src/` from earlier experiments is not included.

## Datasets

MNIST (LeCun et al., 1998), Fashion-MNIST (Xiao et al., 2017) and CIFAR-10 (Krizhevsky, 2009) are downloaded through torchvision and are not redistributed here. PAMAP2 (Reiss and Stricker, 2012) is available from the UCI Machine Learning Repository; download it yourself and follow its terms of use. Each dataset keeps its own licence.

## Acknowledgements

This code re-implements, from the papers, methods proposed by others; no code from their repositories is included.

- **HealSplit** (Xie and Lyu, AAAI 2026): the topology-aware detector and the generator-based healing system used as a comparison (arm A, and the generator in arm B).
- **DALSE** (Chang et al., 2026): the drift-aware latent state estimation that our healer adapts from per-client weight drift to per-class smashed-data drift.
- **BAGAN-GP** (after Huang and Jafari, "Enhanced Balancing GAN"; gradient-penalty critic after Gulrajani et al., NeurIPS 2017): the class-conditioned generator in arm F, implemented from its core mechanism and not as a full reproduction.
- **Generative adversarial networks** (Goodfellow et al., NeurIPS 2014): the standard non-saturating GAN objective used by the generator-based healer.
- **Logit adjustment** (Menon et al., ICLR 2021), **class-balanced loss** (Cui et al., CVPR 2019), **elastic weight consolidation** (Kirkpatrick et al., PNAS 2017) and its use against class imbalance in MConFL (Yuan et al., 2026): the class-imbalance baselines and components.
- **Supervised contrastive learning** (Khosla et al., NeurIPS 2020) and FedIIC (Wu et al., MICCAI 2023): the contrastive refinement step.
- **Federated and split learning**: FedAvg (McMahan et al., 2017), split learning (Vepakomma et al., 2018) and SplitFed (Thapa et al., AAAI 2022).

The smashed-data poisoning attack follows the existing literature; we introduce no new attack.

## Licence
MIT
