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
    
    

if __name__ == '__main__':
    main()