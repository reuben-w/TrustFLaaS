r"""
Retest launcher (supervisor feedback, 10 Oct 2026). NEW file: changes nothing else. Put it in sfl_healsplit\ next to run_battery.py.

It reuses run_battery.py unchanged and adds three arms for the NO-ATTACK comparison with TrustFLaaS (arm M):
  N  logit-adjusted loss        plain split FL, no attack, nothing flagged, no healing, no contrastive step
  O  class-balanced loss        as N, with cb_loss_beta = 0.999 (Cui et al. 2019)
  P  logit-adjusted loss + EWC  as N, with ewc_lambda = 10 from round 5 (the MConFL setting used in arm H)

"Plain split FL": the detector threshold is set so high (--detection-client-mad-tau 1e9) that no client is ever
flagged, so every batch trains normally with the chosen loss. These arms use the same training code as arms C-M,
so only the loss differs. Arm M (TrustFLaaS, no attack) and arm V (no defence, no attack) already exist in final_results
and are NOT rerun. The attacked comparison (TrustFLaaS = I, GAN-based healing = A, BAGAN-GP = F, DALSE only = J)
also needs no new runs: precision and recall come from the confusion matrix each of those runs already saved.

Run from the folder that holds sfl_healsplit\ and src\ , once per partition (120 runs in total: 3 arms x 4 datasets x 2 x 5 seeds):
  python -m sfl_healsplit.run_retest --configs <mnist cfg> <fmnist cfg> <cifar cfg> <pamap2 cfg> --out final_results\retest --device cuda --arms NOP --seeds 42 43 44 45 46 --split iid
  python -m sfl_healsplit.run_retest --configs <same four> --out final_results\retest --device cuda --arms NOP --seeds 42 43 44 45 46 --split noniid
Add --dry-run first to print the commands. Finished runs are skipped, so it can be stopped and resumed.
"""
from . import run_battery as rb

rb.ARMS.update({
    "N": dict(detector="client_shift", detection_client_mad_tau=1e9, healing_mode="dalse",
              la_loss_psi=1.0, attack=False),
    "O": dict(detector="client_shift", detection_client_mad_tau=1e9, healing_mode="dalse",
              la_loss_psi=0.0, cb_loss_beta=0.999, attack=False),
    "P": dict(detector="client_shift", detection_client_mad_tau=1e9, healing_mode="dalse",
              la_loss_psi=1.0, ewc_lambda=10.0, ewc_start_round=5, attack=False),
})

if __name__ == "__main__":
    rb.main()
