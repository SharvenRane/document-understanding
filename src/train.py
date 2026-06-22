"""Training and evaluation helpers for the layout token classifier."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .data import make_dataset, set_global_seed
from .model import LayoutTokenClassifier


def evaluate(
    model: LayoutTokenClassifier,
    input_ids: torch.Tensor,
    bbox: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute token level accuracy over the non padded positions.

    Returns overall accuracy and value accuracy, where "value" excludes the
    background ("other") and "key" classes so it isolates the fields whose
    disambiguation depends on position.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, bbox, mask)
        preds = logits.argmax(dim=-1)

    valid = labels != -100
    correct = (preds == labels) & valid
    overall = correct.sum().item() / max(valid.sum().item(), 1)

    # Value tokens: labels >= 1 and != key. key is the last field id.
    from .data import FIELD_TO_ID  # local import to avoid cycle at module load

    other_id = FIELD_TO_ID["other"]
    key_id = FIELD_TO_ID["key"]
    is_value = valid & (labels != other_id) & (labels != key_id)
    val_correct = (preds == labels) & is_value
    value_acc = val_correct.sum().item() / max(is_value.sum().item(), 1)

    return {"accuracy": overall, "value_accuracy": value_acc}


def train_model(
    n_train: int = 200,
    n_val: int = 60,
    epochs: int = 60,
    lr: float = 3e-3,
    d_model: int = 64,
    use_position: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, object]:
    """Train a classifier on synthetic forms and return model plus metrics.

    The train and validation splits use disjoint seed ranges so validation
    documents are never seen during training.
    """
    set_global_seed(seed)

    tr_ids, tr_box, tr_lab, tr_mask = make_dataset(n_train, seed=1000)
    va_ids, va_box, va_lab, va_mask = make_dataset(n_val, seed=9000)

    L = tr_ids.shape[1]
    model = LayoutTokenClassifier(
        d_model=d_model, max_len=max(64, L), use_position=use_position
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(tr_ids, tr_box, tr_mask)
        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]), tr_lab.reshape(-1)
        )
        loss.backward()
        opt.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            metrics = evaluate(model, va_ids, va_box, va_lab, va_mask)
            print(
                f"epoch {ep:3d} loss {loss.item():.4f} "
                f"val_acc {metrics['accuracy']:.3f} "
                f"val_value_acc {metrics['value_accuracy']:.3f}"
            )

    val_metrics = evaluate(model, va_ids, va_box, va_lab, va_mask)
    return {
        "model": model,
        "val_metrics": val_metrics,
        "val_data": (va_ids, va_box, va_lab, va_mask),
        "final_loss": float(loss.item()),
    }
