from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn import metrics

def main():
    # From consume Tab
    subscription_id = 'd7f39349-a66b-446e-aba6-0053c2cf1c11'
    resource_group = 'aml-quickstarts-214673'
    workspace_name = 'quick-starts-ws-214673'

    workspace = Workspace(subscription_id, resource_group, workspace_name)

    dataset = Dataset.get_by_name(workspace, name='heartfailure-dds')
    
    ds = dataset.to_pandas_dataframe().dropna()
    
    object_columns = ds.select_dtypes(include=['object']).columns


    labelencoder = LabelEncoder()
    for col in object_columns:
        ds[col] = labelencoder.fit_transform(ds[col])

    y_data = ds.pop("DEATH_EVENT")
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(ds, y_data, test_size=0.2, random_state=24)
    run = Run.get_context()

    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="indicates regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    #use model to predict probability that given y value is 1
    y_pred_proba = model.predict_proba(x_test)[::,1]

    #calculate AUC of model
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    run.log("AUC", np.float(auc))
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/hd-model.joblib') 
    

if __name__ == '__main__':
    main()
