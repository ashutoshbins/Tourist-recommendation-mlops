import logging 
from abc import ABC,abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier


class Model(ABC):
    """Abstract class for all models """

    @abstractmethod
    def train(self,X_train,y_train):
        """
        trains the model 
        Args :
            X_train:  Training data 
            y_train:  Training labels 
        Returns:
            None

        """
        pass

class EnsembleModel(Model):
    """
    Ensemble model 
    """
    def train(self,X_train, y_train):
        """Trains the ensemble model 
        Args:
            X_train: Training data 
            y_train: Training labels 
        Returns:
            None 
        """
        try:
            # Random Forest 1
            rf1 = RandomForestClassifier(random_state=42)
            rf1.fit(X_train, y_train)

            # Random Forest 2
            rf2 = RandomForestClassifier(random_state=99)
            rf2.fit(X_train, y_train)

            # Naive Bayes
            nb = GaussianNB()
            nb.fit(X_train, y_train)

            # Create the Voting Classifier
            voting_clf = VotingClassifier(estimators=[('rf1', rf1),('rf2', rf2),('nb', nb)], voting='soft')

            # Fit the ensemble model
            voting_clf.fit(X_train, y_train)
            return voting_clf
        except Exception  as e :
            logging.error(f"Error  in traning model :{e}")
            raise e  