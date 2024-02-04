import sys
import tensorflow
from tensorflow.keras import preprocessing
import numpy as np
import re
from src.exceptions.exception import CustomException
from src.logging.logger import logging


class DataPreprocessor:
    def __init__(self):
        pass
       
    try:
        logging.info("creating tokenizer")
        def create_tokenizer(self,questions,answers):
            tokenizer = preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts(questions +answers)
            vocab_size = len(tokenizer.word_index) + 1
            print(f'VOCAB SIZE: {vocab_size}')
            return tokenizer, vocab_size
        
    except Exception as e:
        logging.error("Error occured in data preprocessing part")
        raise CustomException(e,sys)