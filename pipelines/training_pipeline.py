from src.components.data_ingestion import DataIngestor

from src.components.data_preprocessor import DataPreprocessor

from src.components.data_transformation import DataTransformer

from src.components.build_model import 
from src.components.model_trainer import model

import os
import sys
from src.logging.logger import logging
from src.exceptions.exception import CustomException
import pandas as pd


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestor()
            questions,answers= data_ingestion.data_ingestion()
            return questions,answers
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_preprocessing(self,questions,answers):
        try:
            preprocessor = DataPreprocessor(questions, answers)
            preprocessed_questions, preprocessed_answers = preprocessor.preprocess()
            return preprocessed_questions,preprocessed_answers
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_data_transformation(self,tokenizer,questions,answers,vocab_size):
        
        try:
            transformer = DataTransformer(tokenizer,questions,answers,vocab_size)
            encoder_input, decoder_input, decoder_output, max_len_q, max_len_a = transformer.transform_data()

            return encoder_input, decoder_input, decoder_output, max_len_q, max_len_a
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer=ModelTrainer()
            model_trainer.initate_model_training(train_arr,test_arr)
        except Exception as e:
            raise CustomException(e,sys)
                
    def start_trainig(self):
        try:
            train_data_path,test_data_path=self.start_data_ingestion()
            train_arr,test_arr=self.start_data_transformation(train_data_path,test_data_path)
            self.start_model_training(train_arr,test_arr)
        except Exception as e:
            raise CustomException(e,sys)