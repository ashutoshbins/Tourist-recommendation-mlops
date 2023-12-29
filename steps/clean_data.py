from zenml import step
from typing_extensions import Annotated
from typing import Tuple
import logging 
import pandas as pd 
import sys
import numpy as np
sys.path.append(r'd:\\Projects\\mlops')
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    """cleans the data and divides it into train and test 
    Args: df : Raw data 
    Returns:
        X_train: training data 
        X_test: testing data 
        y_train: training labels 
        y_test: testing labels 
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Error in cleaning data {e}")
        raise e 
    
