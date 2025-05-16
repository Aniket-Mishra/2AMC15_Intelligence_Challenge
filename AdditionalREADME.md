# Supplementary README file for the current implementation of this codebase

## Running agents

The logic for running agents has changed. The current implementation of train.py mandates adding which agent we want to train as an argument

`python train.py grid_configs/Maze.npy --agent MCAgentOn`

The agent names can be found in `agent_config.json`

In the same config, it is also possible to change different parameters for the agents.

Parameters from the original implementation can still be used, such as --no_gui, etc.

## Multi-Agent RL Experiment

These steps need to be taken to obtain the experiment results as found in the report.

Setup for sweeping hyperparameters across RL agents (Q-Learning, Value-Iteration, Monte-Carlo) on various grids, then collecting all the stats in one tidy CSV.

### Overview

1. **Part 1**: Generate `experimental_table.csv` from hyperparameter specs in `experiment_values.json`.
2. **Part 2**: By running `run_experiments.py`, loop through that table, train & evaluate each combo using the logic in `train.py` + `agent_config.json` + `experiment_values.json`, and save the results to `result_multi_experiment.csv`

To try new values or add new hyperparams, just edit **`experiment_values.json`** and run `run_experiments.py`. Everything else will update automatically.
