# E-valuator
Code for paper _E-valuator: Reliable Agent Verifiers with Sequential Hypothesis Testing_. We build a sequential evaluator that can convert any black-box verifier/agent system into one with statistical guarantees. At deployment time, our system can flag and terminate agent trajectories that are likely to be unsuccessful without access to anything but a verifier's (black-box) scores.

## Install
To start, please install our package:

```bash
pip install e-valuator
```

## Quick start
We provide two demo notebooks (and corresponding datasets) in `demos/notebooks/hotpot_example.ipynb` (corresponding dataset in `data/hotpotqa_w_scores_compressed.csv.gz`) and `demos/notebooks/math_example_tokens.ipynb` (corresponding dataset in `data/math_w_scores_compressed.csv.gz`). These notebooks provide examples of the input data format required and evaluation pipeline. In general, the workflow for e-valuator consists of three parts:

1. **Collect agent trajectories and verifier scores**. We provide an example collection script in `demos/collect_verifier_scores/collect_math_example.py`. The trajectories and scores used to calibrate e-valuator must be stored in a csv file (or similar) with (at least) four columns: (1) uq_problem_idx, a unique identifier for each trajectory, (2) step_idx (or num_steps), indicating the step count of the trajectory thus far, (3) judge_probability, indicating the verifier score for that particular step (could also be real-valued, doesn't have to be in 0-1), and (4) solved, a binary indicator column indicating whether the agent successfully solved the problem or not. The columns need not use exactly these names, but if you use a different naming system, you'll need to mark them appropriately upon initialization of e-valuator.

2. **Fit density ratio estimates on calibration set**. Using the calibration set collected in step (1), we'll fit a stepwise density ratio estimator with a binary classifier.

3. **Apply learned density ratio estimates at test time**. We can then apply our learned density ratios from step (2) at test time for _online_ and _sequential_ monitoring of agent trajectories. In particular, given a (black-box) agent's trajectory and corresponding stepwise verifier scores, we can apply e-valuator to terminate poor trajectories.

To start e-valuator:

```python
import evaluator as e_val
ev = e_val.EValuator(
    model_type="logistic",   
    mt_variant="both",    ## "split" is finite time/empirical version of e-valuator. "anytime" is the anytime-valid version. "both" adds both the finite-time and empirical versions as columns to the test set.
    alphas=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], ## by default, alphas = [0.05] unless otherwise specified, as we do here
    # delta=0.1, ## only used for split version, confidence level in false alarm guarantee. default is 0.1 (meaning 90% confidence in false alarm guarantee)
    delta=0.05,
)
```

You now have an e-valuator object that will fit the density ratio estimate using logistic regression. It will find valid thresholds for both the anytime-valid variant and empirical versions of e-valuator, using the list of $\alpha$ values provided. $\delta$ indicates the confidence in the false alarm guarantee. Specifically, $\delta=0.05$ indicates 95\% confidence in the guarantee (see Proposition 3 of the paper for details).

You'll then need to fit e-valuator on a calibration dataframe that has the columns we described above:

```python
ev.fit(cal_df)
```

To then apply it at test-time, you can run:

```python
test_df_with_evals = ev.apply(test_df)
```

## Online Evaluation/Monitoring of Agents
_E-valuator_ assumes black-box access to the verifier/agent system. As such, we do note provide code to directly intervene in any particular agent/verifier system. To deploy _e-valuator_ online, we recommend updating the test_df after each agent action/verifier score:

```python
## assume ev.fit(cal_df) has been called before
x = 0
while agent_not_done:
    ## pseudocode to get a verifier score and add to test df
    verifier_score = verify(partial_traj_step_x)
    test_df.loc[len(test_df)] = ['problem_new', verifier_score1, x]
    test_df_scored = ev.apply(test_df)

    if test_df_scored['anytime_eval'] > 1/alpha:
        terminate
    else:
        x += 1
```
This is inefficient, and if there's interest, we can add better support for online evaluation/monitoring of agents with _e-valuator_.


## Citation
If you use this code, please cite our work.

