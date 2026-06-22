"""Synthetic form documents with tokens, bounding boxes, and field labels.

Each document is a small invoice style form. The form has a fixed set of
fields. Every field occupies a row: a key word sits on the left, a value sits
on the right. Tokens carry text plus a bounding box (x0, y0, x1, y1) on a
normalised 0..1000 page, mirroring the LayoutLM convention.

The learning signal is deliberately split between two channels. The text of a
token is only weakly informative for several fields because the same word
("0", "John", a date fragment) appears in different rows. The vertical
position of a token is what disambiguates which field a value belongs to.
A model that ignores position cannot do better than chance on the value
tokens, so the position channel genuinely matters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


# The set of fields the model classifies each token into. "other" is the
# background class for tokens that do not belong to any field value.
FIELD_NAMES: List[str] = [
    "other",
    "invoice_no",
    "date",
    "name",
    "total",
    "key",
]
NUM_FIELDS: int = len(FIELD_NAMES)
FIELD_TO_ID = {name: i for i, name in enumerate(FIELD_NAMES)}

# A tiny fixed vocabulary. Index 0 is padding, index 1 is the unknown token.
PAD_ID = 0
UNK_ID = 1

_KEY_WORDS = [
    "invoice",
    "no",
    "date",
    "name",
    "total",
    "bill",
    "to",
    "amount",
]
_NAME_WORDS = ["john", "mary", "alex", "sara", "omar", "lena", "raj", "nia"]
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun"]
_DIGITS = [str(d) for d in range(10)]
_CURRENCY = ["$", "usd"]

VOCAB: List[str] = (
    ["<pad>", "<unk>"]
    + _KEY_WORDS
    + _NAME_WORDS
    + _MONTHS
    + _DIGITS
    + _CURRENCY
)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE: int = len(VOCAB)


def encode_token(text: str) -> int:
    """Map a token string to its vocabulary id, falling back to UNK."""
    return TOKEN_TO_ID.get(text.lower(), UNK_ID)


@dataclass
class Form:
    """One synthetic document.

    tokens : list of token strings
    boxes  : list of (x0, y0, x1, y1) integer boxes in 0..1000
    labels : list of field ids, one per token
    """

    tokens: List[str]
    boxes: List[Tuple[int, int, int, int]]
    labels: List[int]

    def __len__(self) -> int:
        return len(self.tokens)


# Each value bearing field lives on its own row. Rows are ordered top to
# bottom on the page. The y band of a row is what tells the model which
# field a value token belongs to.
_ROW_FIELDS = ["invoice_no", "date", "name", "total"]
_ROW_KEY_TEXT = {
    "invoice_no": ["invoice", "no"],
    "date": ["date"],
    "name": ["name"],
    "total": ["total"],
}


def _value_tokens(rng: random.Random, field: str) -> List[str]:
    """Generate the value side token text for a given field.

    The value text is deliberately ambiguous across fields. invoice_no, date
    and total are all made of bare digit tokens, so a digit on its own does
    not reveal which field it belongs to. Only the row (the vertical box
    position) disambiguates them. name is the one field with its own word
    vocabulary, which keeps an anchor the model can learn from text alone.
    """
    if field == "invoice_no":
        return [rng.choice(_DIGITS) for _ in range(3)]
    if field == "date":
        return [rng.choice(_DIGITS) for _ in range(3)]
    if field == "name":
        return [rng.choice(_NAME_WORDS)]
    if field == "total":
        return [rng.choice(_DIGITS) for _ in range(3)]
    raise ValueError(field)


def make_form(seed: int) -> Form:
    """Build a single synthetic form deterministically from a seed."""
    rng = random.Random(seed)

    tokens: List[str] = []
    boxes: List[Tuple[int, int, int, int]] = []
    labels: List[int] = []

    n_rows = len(_ROW_FIELDS)
    # Vertical layout: split the page into evenly spaced row bands.
    top_margin = 80
    bottom_margin = 80
    usable = 1000 - top_margin - bottom_margin
    row_height = usable // n_rows

    for r, field in enumerate(_ROW_FIELDS):
        y0 = top_margin + r * row_height + 5
        y1 = y0 + row_height - 10

        # Key tokens on the left half of the row.
        x = 60
        for kw in _ROW_KEY_TEXT[field]:
            w = 60 + 8 * len(kw)
            tokens.append(kw)
            boxes.append((x, y0, x + w, y1))
            labels.append(FIELD_TO_ID["key"])
            x += w + 15

        # Value tokens on the right half of the row.
        x = 520
        for vt in _value_tokens(rng, field):
            w = 40 + 8 * len(vt)
            tokens.append(vt)
            boxes.append((x, y0, x + w, y1))
            labels.append(FIELD_TO_ID[field])
            x += w + 15

    # Shuffle the reading order of the tokens. The bounding box still encodes
    # where each token sits on the page, but the sequence index no longer
    # reveals the field. This forces the model to rely on the box channel
    # rather than memorising a fixed token order.
    order = list(range(len(tokens)))
    rng.shuffle(order)
    tokens = [tokens[i] for i in order]
    boxes = [boxes[i] for i in order]
    labels = [labels[i] for i in order]

    return Form(tokens=tokens, boxes=boxes, labels=labels)


def form_to_tensors(
    form: Form, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a Form to padded tensors.

    Returns input_ids (L,), bbox (L, 4) float in 0..1, labels (L,) and an
    attention mask (L,) where padded positions are 0. Labels at padded
    positions are set to -100 so they are ignored by cross entropy.
    """
    L = max_len
    input_ids = torch.full((L,), PAD_ID, dtype=torch.long)
    bbox = torch.zeros((L, 4), dtype=torch.float32)
    labels = torch.full((L,), -100, dtype=torch.long)
    mask = torch.zeros((L,), dtype=torch.float32)

    n = min(len(form), L)
    for i in range(n):
        input_ids[i] = encode_token(form.tokens[i])
        x0, y0, x1, y1 = form.boxes[i]
        bbox[i] = torch.tensor([x0, y0, x1, y1], dtype=torch.float32) / 1000.0
        labels[i] = form.labels[i]
        mask[i] = 1.0

    return input_ids, bbox, labels, mask


def make_dataset(
    n_forms: int, seed: int = 0, max_len: int = 24
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batched dataset of n_forms synthetic documents.

    Returns stacked tensors input_ids (N, L), bbox (N, L, 4), labels (N, L)
    and mask (N, L).
    """
    forms = [make_form(seed + i) for i in range(n_forms)]
    longest = max(len(f) for f in forms)
    L = max(max_len, longest)

    ids_list, bbox_list, lab_list, mask_list = [], [], [], []
    for f in forms:
        ids, bbox, lab, mask = form_to_tensors(f, L)
        ids_list.append(ids)
        bbox_list.append(bbox)
        lab_list.append(lab)
        mask_list.append(mask)

    return (
        torch.stack(ids_list),
        torch.stack(bbox_list),
        torch.stack(lab_list),
        torch.stack(mask_list),
    )


def set_global_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
