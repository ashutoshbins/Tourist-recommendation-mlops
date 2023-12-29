import logging
from zenml import step
import pandas as pd
import sys
import mlflow.sklearn
import numpy as np 
sys.path.append(r'd:\\Projects\\mlops')
from src.evaluation import Accuracy  # Assuming Accuracy is the name of your accuracy class
from sklearn.base import ClassifierMixin
remote_server_uri = "http://127.0.0.1:8080" 
mlflow.set_tracking_uri(remote_server_uri)
from zenml.client import Client 
experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> float:
    """
    Evaluates the model on ingested data.

    Args:
        model: Trained model.
        X_test: Testing data.
        y_test: Testing labels.

    Returns:
        float: Evaluation score.
    """
    try:
        predictions = model.predict(X_test)
        accuracy_class = Accuracy()  # Instantiate your accuracy class here
        accuracy = accuracy_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("Accuracy",accuracy)
        return accuracy
    except Exception as e:
        logging.error("Error in Evaluating model: {}".format(e))
        raise e
