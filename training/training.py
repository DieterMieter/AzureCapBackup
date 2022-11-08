import argparse
import os

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core import Dataset

import joblib


def main():

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="indicates regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations")

    args = parser.parse_args()
    
    run = Run.get_context()

    if run.identity.startswith('OfflineRun'):
        ws = Workspace.from_config()

        experiment_name = 'heartfailure-exp'
        experiment = Experiment(ws, experiment_name)

        interactive_run = experiment.start_logging()
    else:
        ws = run.experiment.workspace


    ds = Dataset.get_by_name(ws, name='heartfailure-dds')


    x_df = ds.to_pandas_dataframe().dropna()
    y_df = x_df.pop("DEATH_EVENT")

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, stratify=y_df, random_state=223)
    x_train.reset_index(inplace=True, drop=True)
    x_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    # Make predictions for the test set
    y_pred_test = model.predict(x_test)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
      
    # Save model as -pkl file to the outputs/ folder to use outside the script
    OUTPUT_DIR='./outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_file_name = 'heart_failure_hyperdrive.pkl'
    joblib.dump(value=model, filename=os.path.join(OUTPUT_DIR, model_file_name))

if __name__ == '__main__':
    main()
