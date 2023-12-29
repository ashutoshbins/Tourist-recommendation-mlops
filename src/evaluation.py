import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score

class Evaluation(ABC):
    """Abstract class defining strategy for evaluating our models."""
    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate the score for the model.

        Args:
            y_true: True Labels.
            y_pred: Predicted labels.

        Returns:
            None.
        """
        pass

class Accuracy(Evaluation):
    """Evaluation strategy that uses accuracy function."""
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("Accuracy: {}".format(accuracy))
            return accuracy
        except Exception as e:
            logging.error("Error in calculating Accuracy: {}".format(e))
            raise e
