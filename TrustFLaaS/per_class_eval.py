"""
Per-class test accuracy logging (added for the attack-scenario battery).

Why this exists as its OWN module, not a change to engine.evaluate():
- The main `<run>_metrics.csv` must keep reporting ONLY overall test accuracy
  (that's what the paper's main table and every earlier analysis script read),
  so its schema is deliberately left untouched.
- Per-class accuracy and the confusion matrix go to SEPARATE files next to it:
      <run>_per_class_accuracy.csv   one row per round (round 0 = before training)
      <run>_confusion_matrix.json    final round only (rows = true class,
                                     cols = predicted class)
- Running this after engine.evaluate() consumes NO random numbers (eval mode,
  no_grad, unshuffled test loader), so it cannot change any seed's results.
  Models' train/eval mode is restored afterwards for the same reason.

Class indices are the REMAPPED indices (0..num_classes-1) the rest of the
pipeline uses -- for restricted datasets this is the position in
--restrict-classes (CIFAR-10 frog = 6, MNIST digit 8 = 8, Fashion-MNIST shirt = 6);
for PAMAP2 it is the dataset's own compact activity index.
"""
import csv
import json
import os
from typing import List

import torch


def _accumulate(conf: List[List[int]], preds: List[int], labels: List[int],
                num_classes: int) -> None:
    """conf[true][pred] += 1. Pure Python so it is trivially testable."""
    for p, t in zip(preds, labels):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf[t][p] += 1


@torch.no_grad()
def evaluate_confusion(client_model, server_model, test_loader, device,
                       num_classes: int) -> List[List[int]]:
    """Returns the num_classes x num_classes confusion matrix on the test set."""
    client_was_training = client_model.training
    server_was_training = server_model.training
    client_model.eval()
    server_model.eval()
    conf = [[0] * num_classes for _ in range(num_classes)]
    try:
        for x, y in test_loader:
            x = x.to(device)
            logits = server_model(client_model(x))
            preds = logits.argmax(dim=1).cpu().tolist()
            _accumulate(conf, preds, y.cpu().tolist(), num_classes)
    finally:
        client_model.train(client_was_training)
        server_model.train(server_was_training)
    return conf


class PerClassLogger:
    def __init__(self, metrics_path: str, num_classes: int):
        if metrics_path.endswith("_metrics.csv"):
            stem = metrics_path[: -len("_metrics.csv")]
        else:
            stem = os.path.splitext(metrics_path)[0]
        self.csv_path = stem + "_per_class_accuracy.csv"
        self.confusion_path = stem + "_confusion_matrix.json"
        self.num_classes = num_classes
        header = ["round", "overall_acc"]
        for i in range(num_classes):
            header += [f"class{i}_correct", f"class{i}_total", f"class{i}_acc"]
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def log(self, round_num: int, client_model, server_model, test_loader,
            device, final: bool = False) -> None:
        conf = evaluate_confusion(client_model, server_model, test_loader,
                                  device, self.num_classes)
        grand_total = sum(sum(row) for row in conf)
        grand_correct = sum(conf[i][i] for i in range(self.num_classes))
        row = [round_num, grand_correct / grand_total if grand_total else ""]
        for i in range(self.num_classes):
            total = sum(conf[i])
            correct = conf[i][i]
            row += [correct, total, (correct / total) if total else ""]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if final:
            with open(self.confusion_path, "w") as f:
                json.dump({"round": round_num, "num_classes": self.num_classes,
                           "rows_are": "true_class", "cols_are": "predicted_class",
                           "matrix": conf}, f)
