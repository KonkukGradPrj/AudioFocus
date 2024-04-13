import wandb
import pandas as pd
import numpy as np
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from collections import defaultdict
import os


######################
# Wandb
def collect_runs(project_name):
    api = wandb.Api()   
    
    # Project is specified by <entity/project-name>
    runs = api.runs(project_name)

    return runs


def filter_runs(runs, group_name, exp_names):
    """
    runs: list of wandb runs
    group_name: str
    exp_names: list
    """
    filtered_runs = []
    for run in tqdm(runs):
        configs = {k: v for k,v in run.config.items() if not k.startswith('_')}
        
        if len(configs) == 0:
            continue

        run_group_name = configs['group_name']
        run_exp_name = configs['exp_name']

        if run_group_name == group_name:
            if run_exp_name in exp_names:
                filtered_runs.append(run)
    
    return filtered_runs


def convert_runs_to_dataframe(runs, exp_name_dict):
    """
    runs: wandb runs
    exp_name_dict: {'run_exp_name': 'analysis_exp_name'}
        run_exp_name: str, the exp_name recorded in wandb
        analysis_exp_name: str, the exp_name to be called in analysis.ipynb
    """
    exp_name_list, summary_list, config_list = [], [], []
    for run in tqdm(runs): 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v for k,v in run.config.items() if not k.startswith('_')}
        config_list.append(config)

        run_exp_name = config['exp_name']
        analysis_exp_name = exp_name_dict[run_exp_name]
        exp_name_list.append(analysis_exp_name)

    runs_df = pd.DataFrame({
        "run": runs,
        'exp_name': exp_name_list,
        "summary": summary_list,
        "config": config_list,
        })

    return runs_df


def collect_history(runs_df, metrics, n_samples=50000):
    """
    metrics: list
    """
    metric_dict = {metric: [] for metric in metrics}
    valid_indices = {metric: [] for metric in metrics}

    for idx in tqdm(range(len(runs_df))):
        run_df = runs_df.iloc[idx]
        run = run_df['run']
        history = run.history(samples=n_samples)

        # Track if the current run has all the required metrics
        has_all_metrics = True

        for metric in metrics:
            if metric in history.columns:
                metric_history = history[metric].dropna().values
                metric_dict[metric].append(metric_history)
                valid_indices[metric].append(idx)
            else:
                has_all_metrics = False
                break

        # If not all metrics are present, skip adding this run's data
        if not has_all_metrics:
            continue

    # Add each metric's history to the runs_df
    for metric, values in metric_dict.items():
        # Create a new DataFrame for the valid rows and the corresponding metric history
        valid_df = runs_df.iloc[valid_indices[metric]].copy()
        valid_df[metric + '_history'] = values
        runs_df = runs_df.combine_first(valid_df)

    return runs_df
