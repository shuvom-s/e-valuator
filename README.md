# E-valuator
Code for paper _E-valuator: Reliable Agent Verifiers with Sequential Hypothesis Testing_. We build a sequential evaluator that can convert any black-box verifier/agent system into one with statistical guarantees. At deployment time, our system can flag and terminate agent trajectories that are likely to be unsuccessful without access to anything but a verifier's (black-box) scores.

## Install
To start, please install our package:

```bash
pip install e-valuator
```

## Quick start
Once installed, you can boot up e-valuator with `from evaluator import EValuator`. We provide two demo notebooks (and corresponding datasets) in `demos/notebooks/hotpot_example.ipynb` (corresponding dataset in `data/hotpotqa_cleaned_w_scores.csv`) and `demos/notebooks/math_example_tokens.ipynb` (corresponding dataset in `data/math_cleaned_w_scores.csv`).

These notebooks provide examples of the input data format required and evaluation pipeline.

