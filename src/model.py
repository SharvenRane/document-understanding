"""A small LayoutLM style token classifier.

The model embeds three signals for every token and adds them:

  1. the token text, through a word embedding table,
  2. the four bounding box coordinates, through linear position embeddings
     (x0, y0, x1, y1 plus width and height), and
  3. a sequence position embedding.

The combined embedding goes through a shallow Transformer encoder so tokens
can attend to each other, then a linear head classifies each token into a
field. The bounding box channel is what lets the model tell apart value
tokens that share the same text but sit in different rows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import NUM_FIELDS, PAD_ID, VOCAB_SIZE


class LayoutTokenClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        num_fields: int = NUM_FIELDS,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
        use_position: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_position = use_position

        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.seq_pos_emb = nn.Embedding(max_len, d_model)

        # Box embedding: project the 6 box features (x0,y0,x1,y1,w,h) into the
        # model dimension. This is the spatial channel.
        self.box_proj = nn.Linear(6, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_fields)
        self.max_len = max_len

    def _box_features(self, bbox: torch.Tensor) -> torch.Tensor:
        """Turn (x0,y0,x1,y1) into (x0,y0,x1,y1,width,height)."""
        x0, y0, x1, y1 = bbox.unbind(dim=-1)
        w = (x1 - x0).clamp(min=0.0)
        h = (y1 - y0).clamp(min=0.0)
        return torch.stack([x0, y0, x1, y1, w, h], dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per token logits of shape (B, L, num_fields)."""
        B, L = input_ids.shape
        device = input_ids.device

        emb = self.word_emb(input_ids)

        if self.use_position:
            box_feats = self._box_features(bbox)
            emb = emb + self.box_proj(box_feats)

        positions = torch.arange(L, device=device).clamp(max=self.max_len - 1)
        emb = emb + self.seq_pos_emb(positions).unsqueeze(0)

        # TransformerEncoder ignores positions where the padding mask is True.
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = mask < 0.5  # True == padded == ignore

        h = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        return self.head(h)
