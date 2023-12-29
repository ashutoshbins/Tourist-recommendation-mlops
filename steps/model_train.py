import logging 
import pandas as pd 
from zenml import step
import numpy as np
import sys
import mlflow
sys.path.append(r'd:\\Projects\\mlops')
from src.model_dev import EnsembleModel 
from sklearn.ensemble import VotingClassifier
from zenml.client import Client 
remote_server_uri = "http://127.0.0.1:8080" 
mlflow.set_tracking_uri(remote_server_uri)
experiment_tracker=Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: np.ndarray,
                X_test: np.ndarray,
                y_train: np.ndarray,
                y_test:np.ndarray,
                model_name: str = "EnsembleModel") -> VotingClassifier:
    '''Trains the model on ingested data 
    Args: 
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
        model_name: Name of the model to train (default is "EnsembleModel")
    Returns:
        Classifier model
    '''
    try:
        model = None
        if model_name == "EnsembleModel":
            mlflow.sklearn.autolog()
            model = EnsembleModel().train(X_train, y_train)
            return model 
        else:
            raise ValueError("Model {} not supported".format(model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e
