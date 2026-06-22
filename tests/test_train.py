"""Behavioural tests: position matters and classification beats chance."""

import torch

from src.data import FIELD_TO_ID
from src.train import evaluate, train_model


def test_classification_beats_chance_on_value_tokens():
    # invoice_no, date and total are all digit valued, so guessing by text
    # alone is roughly one in three on those fields. A trained layout aware
    # model should land far above that floor.
    out = train_model(epochs=120, use_position=True, seed=0)
    value_acc = out["val_metrics"]["value_accuracy"]
    chance = 1.0 / 3.0
    assert value_acc > chance + 0.3, f"value_acc={value_acc}"
    # And it should be strongly above chance overall too.
    assert out["val_metrics"]["accuracy"] > 0.85


def test_position_helps_over_no_position():
    # The model with bounding boxes should beat the text only model on the
    # ambiguous value fields by a clear margin.
    with_pos = train_model(epochs=120, use_position=True, seed=0)
    no_pos = train_model(epochs=150, use_position=False, seed=0)
    v_pos = with_pos["val_metrics"]["value_accuracy"]
    v_nopos = no_pos["val_metrics"]["value_accuracy"]
    assert v_pos > v_nopos + 0.2, f"pos={v_pos} nopos={v_nopos}"


def test_shuffling_boxes_changes_predictions():
    # If the model genuinely uses position, permuting which box belongs to
    # which token must change a non trivial number of predictions.
    out = train_model(epochs=120, use_position=True, seed=0)
    model = out["model"]
    ids, bbox, lab, mask = out["val_data"]

    model.eval()
    torch.manual_seed(7)
    with torch.no_grad():
        base = model(ids, bbox, mask).argmax(dim=-1)
        perm = torch.randperm(bbox.shape[1])
        shuffled = model(ids, bbox[:, perm, :], mask).argmax(dim=-1)

    valid = lab != -100
    changed = ((base != shuffled) & valid).sum().item()
    total = valid.sum().item()
    assert changed > 0.05 * total, f"changed {changed} of {total}"


def test_shuffling_boxes_lowers_value_accuracy():
    # Beyond merely changing predictions, scrambling the layout should hurt
    # accuracy on the position dependent value tokens.
    out = train_model(epochs=120, use_position=True, seed=0)
    model = out["model"]
    ids, bbox, lab, mask = out["val_data"]

    clean = evaluate(model, ids, bbox, lab, mask)["value_accuracy"]
    torch.manual_seed(3)
    perm = torch.randperm(bbox.shape[1])
    scrambled = evaluate(model, ids, bbox[:, perm, :], lab, mask)[
        "value_accuracy"
    ]
    assert scrambled < clean - 0.2, f"clean={clean} scrambled={scrambled}"


def test_evaluate_returns_fractions():
    out = train_model(epochs=20, use_position=True, seed=0)
    m = evaluate(out["model"], *out["val_data"])
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["value_accuracy"] <= 1.0
