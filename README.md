# Reformer-pytorch
Implements [Reformer: The Efficient Transformer](https://openreview.net/forum?id=rkgNKkHtvB) in pytorch. (Work in progress)

## Prerequisites

- Tested with Python 3.7.5, Pytorch 1.3.1.
- This code is built upon the [pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning/) framework.
- `pip install -r requirements.txt`

## How to train

### Datasets

- Currently only the synthetic dataset described in the paper is implemented.
- Various parameters are editable in `config\config.yaml` under `data:`.

### Running the code

- `python3 trainer.py -c \path\to\config\yaml -n [name of run] -b [batch size] -f [fast dev run] -v [resume version number]
- The `-f` flag is used for debugging; only one batch of training, validation, and testing will be calculated.
- The `-v` flag is used for resuming from checkpoints; leave empty for new version.

## How to sample

### Preparing the checkpoints

- Currently not implemented.

### Running the code

- Currently not implemented.

## To-do

- [x] Implement general framework of Reformer
- [x] Rewrite using [pytorch-lightning](https://github.com/williamFalcon/pytorch-lightning/) framework
- [x] Implement LSH attention
- [ ] Implement reversible layer
- [ ] Implement various datasets
- [ ] Implement sampling

## Implementation Authors

- [June Young Yi](<https://github.com/Rick-McCoy>)

## License

MIT License

## Acknowlegdements

- The general structure of this code is based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html), albeit heavily modified.
- I am aware that [reformer-lm](https://github.com/zbloss/reformer_lm) exists. However, I was frustrated with the original [trax implementation](https://github.com/google/trax/blob/master/trax/models/research/reformer.py) that the authors provided, and decided to rewrite the entire thing from the ground up. Naturally, expect bugs everywhere.
