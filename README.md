EE 641 - HW2 - Cheng Jung, Lin
=========================
Full name: `Cheng Jung, Lin`

USC email address: `clin1650@usc.edu`

# Instructions
## Problem 1
- Run `setup.py` in `problem1/`, make sure `data/fonts/` is in `problem1/`.
- Simply run `train.py`, and the results should be in their folder.

## Problem 2
- Run `setup.py` in `problem2/`, make sure `data/drums/` is in `problem2/`.
- There are three annealing strategies in this model, cyclic is set to default.
- You may run the other two strategies with command.
- `python train.py --run-tag linear --kl-mode linear` for linear.
- `python train.py --run-tag constant --kl-mode constant --free-bits 0.0` for constant.
