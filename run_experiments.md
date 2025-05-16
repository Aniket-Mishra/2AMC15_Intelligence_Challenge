# Multi-Agent RL Experiment

Setup for sweeping hyperparameters across RL agents (Q-Learning, Value-Iteration, Monte-Carlo) on various grids, then collecting all the stats in one tidy CSV.

## Overview

1. **Part 1**: Generate `experimental_table.csv` from hyperparameter specs in `experiment_values.json`.
2. **Part 2**: Loop through that table, train & evaluate each combo using the logic in `train.py` + `agent_config.json` + `experiment_values.json`, and save the results to `result_multi_experiment.csv` inside `experiment_results`

To try new values or add new hyperparams, just edit **`experiment_values.json`** and run `2_generate_experiments.ipynb`. Everything else will update automatically.
