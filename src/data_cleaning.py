import logging
import pandas as pd
import spacy
import string
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

#global variables
exclude=string.punctuation
porter = PorterStemmer()
cv = CountVectorizer(ngram_range=(1,7))

def rem_punc(text:str)->str:
    '''to remove punctuation from a text'''
    return text.translate(str.maketrans('','',exclude))

def tokenize_and_create_string(input_text):
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Tokenize the input text using SpaCy
    tokens = nlp(input_text)

    # Create a new string from the tokens
    new_string = ' '.join([token.text for token in tokens])

    return new_string
def apply_porter_stemmer(text):
    words = word_tokenize(text)
    stemmed_words = [porter.stem(word) for word in words]
    return ' '.join(stemmed_words)

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series,np.ndarray]:
        pass


class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
       
        try:
            data['description'] = data['description'].apply(lambda x:x.lower())
            sw_list = stopwords.words('english') 
            data['description'] = data['description'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
            data['description']=data['description'].apply(rem_punc)
            #data['description']=data['description'].apply(tokenize_and_create_string)
            #data['description'] = data['description'].apply(apply_porter_stemmer)
            
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series,np.ndarray]:
        try:
            X = data.iloc[:,0:1]
            y = data['category']
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
            X_train= cv.fit_transform(X_train['description']).toarray()
            X_test= cv.transform(X_test['description']).toarray()
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
class DataCleaning : 
    '''class for cleaning data which processes the data and divides it into train and test '''
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy
    def handle_data(self)->Union[pd.DataFrame,pd.Series,np.ndarray]:
        '''Handle data '''
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e :
            logging.error(f"Error in handling data: {e} ")
            raise e
if __name__=="__main__":
    data=pd.read_csv(r"D:\Projects\mlops\data\NLP.csv")
    data_divide=DataCleaning(data,DataDivideStrategy())
    print(data_divide.handle_data())
   

            
