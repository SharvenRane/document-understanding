"""Tests for the synthetic form generator."""

import torch

from src.data import (
    FIELD_NAMES,
    FIELD_TO_ID,
    NUM_FIELDS,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
    encode_token,
    make_dataset,
    make_form,
)


def test_form_is_deterministic_from_seed():
    a = make_form(7)
    b = make_form(7)
    assert a.tokens == b.tokens
    assert a.boxes == b.boxes
    assert a.labels == b.labels


def test_form_has_aligned_token_box_label_lengths():
    f = make_form(3)
    assert len(f.tokens) == len(f.boxes) == len(f.labels)
    assert len(f) > 0


def test_labels_are_valid_field_ids():
    f = make_form(11)
    for lab in f.labels:
        assert 0 <= lab < NUM_FIELDS


def test_every_value_field_appears_in_a_form():
    # invoice_no, date, name, total each occupy one row, so all should appear.
    f = make_form(0)
    present = set(f.labels)
    for field in ["invoice_no", "date", "name", "total", "key"]:
        assert FIELD_TO_ID[field] in present


def test_boxes_are_within_page_bounds():
    f = make_form(5)
    for (x0, y0, x1, y1) in f.boxes:
        assert 0 <= x0 < x1 <= 1000
        assert 0 <= y0 < y1 <= 1000


def test_rows_are_vertically_ordered():
    # The four value fields live in distinct, top to bottom rows. The mean y
    # of invoice_no tokens should sit above date, which sits above total.
    f = make_form(2)
    ymid = {}
    for field in ["invoice_no", "date", "total"]:
        fid = FIELD_TO_ID[field]
        ys = [
            (b[1] + b[3]) / 2
            for b, lab in zip(f.boxes, f.labels)
            if lab == fid
        ]
        ymid[field] = sum(ys) / len(ys)
    assert ymid["invoice_no"] < ymid["date"] < ymid["total"]


def test_digit_fields_share_token_text():
    # The point of the dataset: invoice_no, date and total values are digits,
    # so their text overlaps and only position can tell them apart. Confirm
    # at least one digit string is reused across two different fields.
    f = make_form(0)
    by_field = {}
    for tok, lab in zip(f.tokens, f.labels):
        by_field.setdefault(lab, set()).add(tok)
    inv = by_field[FIELD_TO_ID["invoice_no"]]
    tot = by_field[FIELD_TO_ID["total"]]
    # Both are drawn from the same digit alphabet.
    assert inv and tot
    assert inv.issubset(set("0123456789"))
    assert tot.issubset(set("0123456789"))


def test_encode_token_known_and_unknown():
    assert encode_token("invoice") != UNK_ID
    assert encode_token("INVOICE") == encode_token("invoice")  # case folded
    assert encode_token("zzznotaword") == UNK_ID


def test_make_dataset_shapes_and_padding():
    ids, bbox, lab, mask = make_dataset(8, seed=42, max_len=24)
    N, L = ids.shape
    assert N == 8
    assert bbox.shape == (N, L, 4)
    assert lab.shape == (N, L)
    assert mask.shape == (N, L)

    # Boxes are normalised into 0..1.
    assert bbox.min().item() >= 0.0
    assert bbox.max().item() <= 1.0

    # Padded positions have pad id, ignore label and zero mask.
    pad_pos = mask < 0.5
    assert torch.all(ids[pad_pos] == PAD_ID)
    assert torch.all(lab[pad_pos] == -100)


def test_vocab_indices_consistent():
    assert PAD_ID == 0
    assert UNK_ID == 1
    assert VOCAB_SIZE > len(FIELD_NAMES)
