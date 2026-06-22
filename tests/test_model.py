"""Tests for the LayoutTokenClassifier forward behaviour."""

import torch

from src.data import NUM_FIELDS, make_dataset
from src.model import LayoutTokenClassifier


def test_forward_output_shape():
    ids, bbox, lab, mask = make_dataset(4, seed=1, max_len=20)
    model = LayoutTokenClassifier(max_len=64)
    logits = model(ids, bbox, mask)
    assert logits.shape == (ids.shape[0], ids.shape[1], NUM_FIELDS)


def test_logits_are_finite():
    ids, bbox, lab, mask = make_dataset(4, seed=1, max_len=20)
    model = LayoutTokenClassifier(max_len=64)
    logits = model(ids, bbox, mask)
    assert torch.isfinite(logits).all()


def test_box_channel_changes_output_when_enabled():
    # With position enabled, perturbing the boxes must change the logits.
    torch.manual_seed(0)
    ids, bbox, lab, mask = make_dataset(4, seed=1, max_len=20)
    model = LayoutTokenClassifier(max_len=64, use_position=True)
    model.eval()
    with torch.no_grad():
        base = model(ids, bbox, mask)
        moved = model(ids, bbox + 0.3, mask)
    assert not torch.allclose(base, moved, atol=1e-5)


def test_box_channel_ignored_when_disabled():
    # With position disabled, the box input must not affect the output.
    torch.manual_seed(0)
    ids, bbox, lab, mask = make_dataset(4, seed=1, max_len=20)
    model = LayoutTokenClassifier(max_len=64, use_position=False)
    model.eval()
    with torch.no_grad():
        base = model(ids, bbox, mask)
        moved = model(ids, bbox + 0.5, mask)
    assert torch.allclose(base, moved, atol=1e-6)


def test_backward_produces_gradients():
    ids, bbox, lab, mask = make_dataset(4, seed=1, max_len=20)
    model = LayoutTokenClassifier(max_len=64)
    logits = model(ids, bbox, mask)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, NUM_FIELDS), lab.reshape(-1), ignore_index=-100
    )
    loss.backward()
    # The box projection should receive gradient, confirming it is wired in.
    assert model.box_proj.weight.grad is not None
    assert model.box_proj.weight.grad.abs().sum().item() > 0
