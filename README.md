# document-understanding

Layout aware key value extraction for documents, in the spirit of LayoutLM but
small enough to train on a CPU in seconds. Every token in a document carries
two things: the word itself and a bounding box telling you where it sits on the
page. This project shows that the box position is not decoration. It is the
signal that lets a model decide which field a value belongs to when the text
alone is ambiguous.

## The idea

Take a simple invoice style form. It has an invoice number row, a date row, a
name row, and a total row. The number "7" might be part of the invoice number,
part of the date, or part of the total. Reading the digit gives you nothing.
What tells you the field is the row the digit lives in, which is to say its
vertical position on the page.

The model embeds three things for each token and adds them together: the word
through an embedding table, the four bounding box coordinates (plus width and
height) through a linear projection, and the sequence index. A shallow
Transformer encoder lets tokens attend to one another, and a linear head
predicts a field label per token. This mirrors how LayoutLM folds 2D position
into the input embeddings.

## The synthetic data

There are no downloads. `src/data.py` generates forms on the fly from a seed,
so the data is deterministic and tiny. The generator does two things on
purpose to make position the load bearing signal:

1. The invoice number, date, and total are all made of bare digit tokens. A
   lone digit therefore does not reveal its field. Only the row it sits in
   does.
2. The reading order of the tokens inside each form is shuffled. That removes
   the shortcut of memorising a fixed token order, so the sequence index by
   itself no longer leaks the layout. The bounding box becomes the reliable
   cue.

Each field is labelled per token: `invoice_no`, `date`, `name`, `total`, the
`key` words on the left of each row, and `other` for background.

## Layout

```
src/data.py    synthetic form generator, vocabulary, tensor packing
src/model.py   the small LayoutLM style token classifier
src/train.py   training loop and token level evaluation
tests/         pytest behaviour tests
```

## Running it

Install the dependencies and train a model:

```
pip install -r requirements.txt
python -c "from src.train import train_model; print(train_model(verbose=True)['val_metrics'])"
```

The validation split uses a disjoint seed range from training, so the model is
graded on forms it never saw.

## What the tests check

The tests are behaviour checks, not snapshots of a number. They assert the
properties that make this a real layout model:

- Field classification beats chance on the ambiguous digit fields, where
  guessing from text alone would sit near one in three.
- The model trained with bounding boxes beats the same model trained without
  them, on exactly those ambiguous fields.
- Permuting which box belongs to which token changes a meaningful fraction of
  predictions, and lowers value accuracy. If the model ignored position this
  would do nothing.
- When position is switched off, perturbing the boxes leaves the output
  identical, confirming the box channel is genuinely the only path for spatial
  information.

Run them with:

```
python -m pytest tests/ -q
```

On this machine the full suite passes (20 tests). A trained model with the
default settings reaches full value accuracy on the held out forms, while the
text only variant lands far below it, around the chance floor for the digit
fields. Scrambling the boxes collapses the value accuracy back down, which is
the whole point.

## Notes

This is a teaching scale implementation. The vocabulary is a fixed list of a
few dozen tokens, the encoder is two layers wide, and the documents are a
handful of rows. The mechanics are the same ones LayoutLM uses at scale:
fold 2D position into the token embeddings and let attention do the rest.
