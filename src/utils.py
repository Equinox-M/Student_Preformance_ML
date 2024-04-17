import os 
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Function to save Python object to a file.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj: Python object to be saved.
        
    Raises:
        CustomException: If an error occurs during object saving.

    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Function to evaluate multiple models using grid search cross-validation and return the evaluation report.

    Args:
        X_train (numpy.ndarray): Features of the training data.
        y_train (numpy.ndarray): Target variable of the training data.
        X_test (numpy.ndarray): Features of the test data.
        y_test (numpy.ndarray): Target variable of the test data.
        models (dict): Dictionary containing model names as keys and corresponding model objects as values.
        param (dict): Dictionary containing model names as keys and hyperparameter grids as values.

    Returns:
        dict: Evaluation report containing model names as keys and R-squared scores on the test data as values.

    Raises:
        CustomException: If an error occurs during model evaluation.

    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)    