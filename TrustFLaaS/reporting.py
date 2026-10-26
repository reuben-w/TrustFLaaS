"""
Transparency reporting layer, sitting between Detect and Generate in the
detect > report > generate > validate > heal pipeline.

Design rationale (see project notes for the fuller argument): a client-private
report alone can't reveal a *systemic* pattern -- a client whose data is
disadvantaged every round has no way to distinguish that from ordinary bad
luck without a comparison point. So this produces two views every round:
  - a PRIVATE, per-client report (their own cumulative outcomes only)
  - a PUBLIC, per-class aggregate (anonymized to the class level, visible to
    everyone) -- this is the view that would have made our own frog finding
    visible without needing to build dedicated diagnostic instrumentation
    after the fact.

Two touchpoints per round, matching where information actually becomes known:
  - record_detection(...): called right after Detect, before Generate/Validate/
    Heal have run. Reports THIS round's raw flagged counts, plus CUMULATIVE
    stats as of the end of the PREVIOUS round (this round's healing outcome
    isn't known yet at this point in the pipeline).
  - record_outcome(...): called after Heal. Updates the running cumulative
    totals with this round's actual replaced/dropped outcome, so NEXT round's
    report reflects it.

A single-round success rate is noisy (small counts swing widely) -- this is
deliberately why the report always surfaces the CUMULATIVE rate, not just
this round's raw numbers.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ClientClassStats:
    total: int = 0
    flagged: int = 0
    replaced: int = 0
    dropped: int = 0

    @property
    def success_rate(self) -> Optional[float]:
        return self.replaced / self.flagged if self.flagged > 0 else None


class TransparencyLog:
    """Persistent, round-over-round running log. One instance per experiment run."""

    def __init__(self, num_classes: int, num_clients: int,
                 class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.class_names = class_names or [str(c) for c in range(num_classes)]
        self.cumulative_by_class: Dict[int, ClientClassStats] = {
            c: ClientClassStats() for c in range(num_classes)
        }
        self.cumulative_by_client_class: Dict[Tuple[int, int], ClientClassStats] = {}
        self.round_reports: List[dict] = []

    def _get_client_class(self, client_id: int, class_idx: int) -> ClientClassStats:
        key = (client_id, class_idx)
        if key not in self.cumulative_by_client_class:
            self.cumulative_by_client_class[key] = ClientClassStats()
        return self.cumulative_by_client_class[key]

    def record_detection(self, round_num: int, y: torch.Tensor, flagged: torch.Tensor,
                          client_ids: torch.Tensor) -> dict:
        this_round_flagged_by_class = {
            c: int(((y == c) & flagged).sum().item()) for c in range(self.num_classes)
        }

        aggregate_report = {}
        for c in range(self.num_classes):
            stats = self.cumulative_by_class[c]
            aggregate_report[self.class_names[c]] = {
                "flagged_this_round": this_round_flagged_by_class[c],
                "cumulative_success_rate": stats.success_rate,
                "cumulative_flagged": stats.flagged,
                "cumulative_replaced": stats.replaced,
                "cumulative_dropped": stats.dropped,
            }

        private_reports = {}
        for cid in sorted(set(client_ids.tolist())):
            client_mask = client_ids == cid
            client_report = {}
            for c in range(self.num_classes):
                class_mask = client_mask & (y == c)
                flagged_now = int((class_mask & flagged).sum().item())
                stats = self._get_client_class(cid, c)
                client_report[self.class_names[c]] = {
                    "flagged_this_round": flagged_now,
                    "your_cumulative_success_rate": stats.success_rate,
                    "your_cumulative_flagged": stats.flagged,
                }
            private_reports[int(cid)] = client_report

        report = {"round": round_num, "aggregate": aggregate_report,
                   "private_by_client": private_reports}
        self.round_reports.append(report)
        return report

    def record_outcome(self, y: torch.Tensor, flagged: torch.Tensor,
                        client_ids: torch.Tensor, keep_mask: torch.Tensor) -> None:
        """keep_mask[i] True = sample i was successfully healed (replaced), for
        flagged samples. flagged & not keep_mask = dropped."""
        for i in range(len(y)):
            c, cid = int(y[i].item()), int(client_ids[i].item())
            is_flagged = bool(flagged[i].item())

            self.cumulative_by_class[c].total += 1
            client_stats = self._get_client_class(cid, c)
            client_stats.total += 1

            if is_flagged:
                self.cumulative_by_class[c].flagged += 1
                client_stats.flagged += 1
                if bool(keep_mask[i].item()):
                    self.cumulative_by_class[c].replaced += 1
                    client_stats.replaced += 1
                else:
                    self.cumulative_by_class[c].dropped += 1
                    client_stats.dropped += 1

    def write_round_report(self, output_dir: str, round_num: int) -> None:
        """output_dir should be the run's own results directory (e.g.
        os.path.dirname(results_path) in defense.py) -- already unique per
        run, so reports land at <output_dir>/transparency_reports/..."""
        report = self.round_reports[-1]
        assert report["round"] == round_num, "call write_round_report right after record_detection"

        agg_dir = os.path.join(output_dir, "transparency_reports", "aggregate")
        os.makedirs(agg_dir, exist_ok=True)
        with open(os.path.join(agg_dir, f"round_{round_num:03d}.json"), "w") as f:
            json.dump({"round": round_num, "aggregate": report["aggregate"]}, f, indent=2)

        for cid, client_report in report["private_by_client"].items():
            client_dir = os.path.join(output_dir, "transparency_reports", f"client_{cid}")
            os.makedirs(client_dir, exist_ok=True)
            with open(os.path.join(client_dir, f"round_{round_num:03d}.json"), "w") as f:
                json.dump({"round": round_num, "report": client_report}, f, indent=2)
