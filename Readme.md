# Heart Failure Prediction

In this project I trained and deployed a model to predict if someone will die of heart failure based on risk factors.

## Dataset

### Overview
I used this Dataset from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data?resource=download

It is a Dataset that correlates heart_failure with certain risk factors like age, certain illnesses or parameters from blood tests.

### Task
I will try to predict whether or not a person will die from heart failure based on these features:
| Feature | Explanation | Measurement |
| :---: | :---: | :---: |
| *age* | Age of patient | Years (40-95) |
| *anaemia* | Decrease of red blood cells or hemoglobin | Boolean (0=No, 1=Yes) |
| *creatinine-phosphokinase* | Level of the CPK enzyme in the blood | mcg/L |
| *diabetes* | Whether the patient has diabetes or not | Boolean (0=No, 1=Yes) |
| *ejection_fraction* | Percentage of blood leaving the heart at each contraction | Percentage |
| *high_blood_pressure* | Whether the patient has hypertension or not | Boolean (0=No, 1=Yes) |
| *platelets* | Platelets in the blood | kiloplatelets/mL	|
| *serum_creatinine* | Level of creatinine in the blood | mg/dL |
| *serum_sodium* | Level of sodium in the blood | mEq/L |
| *sex* | Female (F) or Male (M) | Binary (0=F, 1=M) |
| *smoking* | Whether the patient smokes or not | Boolean (0=No, 1=Yes) |
| *time* | Follow-up period | Days |

### Access
I downloaded the Dataset from Kaggle and uploaded it to Datasets in my workspace. From there I can access it using:
```Dataset.get_by_name(ws, name="heartfailure-dds")```

## Automated ML
I chose a timeout of 30min because my overall time for this project is limited. Max concurrent iterations is chosen 1 lower than the amount of nodes of 
the compute cluster as proposed in the exercises. As this is a classification problem I chose AUC which gives a good balance between False Positive and 
False Negative predictions.


### Results
What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
With AutoML I got an AUC of 0.92 which is pretty good. The Model is a VotingEnsemble with 11 voting models. I doubt that a much better result would have been realistic here 
but I could have Done a more extensive sweep of the Hyperparameter Space by giving it more time and disabling early stopping.


![AutoML RunDetails widget](./RunDetailsWidget1.PNG)
![AutoML RunDetails widget](./RunDetailsWidget2.PNG)

![AutoML best model details](./Best_model_Run_Id_new.PNG)

## Result Update
I had to rerun everything to comply with the request in the review because my workspace had already timed out.
This time I got a ExtremeRandomTrees model with a StandardScalerWrapper which reached a AUC of 0.91285 so slightly worse than last time the Accuracy is 0.81586.
The Hyperparameters are:
- Data Transformation:
```
{
    "class_name": "StandardScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {
        "with_mean": false,
        "with_std": true
    },
    "prepared_kwargs": {},
    "spec_class": "preproc"
}
```
- Training Algorithm:
```
{
    "class_name": "ExtraTreesClassifier",
    "module": "sklearn.ensemble",
    "param_args": [],
    "param_kwargs": {
        "bootstrap": false,
        "class_weight": "balanced",
        "criterion": "gini",
        "max_features": null,
        "min_samples_leaf": 0.01,
        "min_samples_split": 0.056842105263157895,
        "n_estimators": 200,
        "oob_score": false
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
```

## Result Update 2:
I managed to get a Voting Ensemble again to comply with the review request to name the inner estimators and how their predictions are aggregated.
AUC is 0.92091 and Accuracy 0.85287.

The VotingEnsemble consists of the following estimators and respective weights with which their prediction is multiplied and then summed:

| Inner Estimator | Ensemble weight |
| :---: | :---: |
| TruncatedSVDWrapper, ExtremeRandomTrees | 0.21428571428571427| 
| StandardScalerWrapper, LightGBM | 0.07142857142857142 |
| StandardScalerWrapper, ExtremeRandomTrees | 0.21428571428571427 |
| MaxAbsScaler, RandomForest | 0.07142857142857142 |
| StandardScalerWrapper, RandomForest | 0.07142857142857142 |
| StandardScalerWrapper, XGBoostClassifier | 0.07142857142857142 |
| MinMaxScaler, ExtremeRandomTrees | 0.07142857142857142 |
| MinMaxScaler, ExtremeRandomTrees | 0.07142857142857142 |
| StandardScalerWrapper, XGBoostClassifier | 0.07142857142857142 |
| SparseNormalizer, RandomForest | 0.07142857142857142 |

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
I chose a Logistic Regression Model because it is very fast to train. I only varied '--C' which affects regularizationand 'max_iter' which affects the
Maximum number of iterations also to save some time.

### Results
What are the results you got with your model? What were the parameters of the model? How could you have improved it?
I achieved an accuracy of 0.76 which is not perfect for comparison as a chose a different metric compared to the AutoML approach. It is still safe to say that 
it is worse than the AutoML model.
The Hyperparameters found to be best by Hyperdrive were:
- C = 1.1528233412444724
- max_iter = 80

I could have improved the result by chosing a more complex model and doing a more extensive sweep of the hyperparameter space. 
The trivial way to do it would be to start with a VotingEnsemble similar to the one chosen by AutoML as I already knew
it worked well

![Hyperdrive RunDetails widget](./Hyperdrive_RunDetails1.PNG)
![Hyperdrive RunDetails widget](./Hyperdrive_RunDetails2.PNG)
![Hyperdrive RunDetails widget](./Hyperdrive_RunDetails3.PNG)
![Hyperdrive RunDetails widget](./Hyperdrive_RunDetails4.PNG)
![Hyperdrive RunDetails widget](./Hyperdrive_RunDetails5.PNG)
![Hyperdrive RunDetails widget](./Hyperdrive_RunDetails1.PNG)

![Hyperdrive best model details](./Best_hypermodel.PNG)

## Model Deployment
Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
You can get all Info about how to query an Endpoint from the "Consume" Tab of the Endpoint:
![Endpoint 1](./Endpoint_use1.PNG)
With the sample Code from Consume Tab you can manipulate the data and query the endpoint like this:
![Endpoint 2](./Endpoint_use2.PNG)

## Screen Recording
https://vimeo.com/768688488

