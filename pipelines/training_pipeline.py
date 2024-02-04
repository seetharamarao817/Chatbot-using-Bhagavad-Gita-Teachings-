from src.components.data_ingestion import DataIngestor

from src.components.data_preprocessor import DataPreprocessor

from src.components.data_transformation import DataTransformer

from src.components.build_model import model

from src.components.model_trainer import ChatbotTrainer

from src.utils.utils import save_tokenizer_to_json
import os
import sys
from src.logging.logger import logging
from src.exceptions.exception import CustomException
import pandas as pd
import json
from keras.preprocessing.text import tokenizer_from_json


class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestor()
            questions,answers= data_ingestion.data_ingestion()
            return questions,answers
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_preprocessing(self, questions, answers):
        try:
            tokenizer_path = 'artifacts/tokenizer.json'

            if os.path.exists(tokenizer_path):
                # Load existing tokenizer
                token = load_tokenizer_from_json(tokenizer_path)
                # Update the existing tokenizer with new data
                token.fit_on_texts(questions + answers)
                vocab_size = len(token.word_index) + 1
            else:
                # Create new tokenizer
                preprocessor = DataPreprocessor()
                token, vocab_size = preprocessor.create_tokenizer(questions, answers)
                # Save the new tokenizer for future use
                save_tokenizer_to_json(token, tokenizer_path)

            return token, vocab_size
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_data_transformation(self,tokenizer,questions,answers,vocab_size):
        
        try:
            transformer = DataTransformer(tokenizer,questions,answers,vocab_size)
            encoder_input, decoder_input, decoder_output, max_len_q, max_len_a = transformer.transform_data()

            return encoder_input, decoder_input, decoder_output, max_len_q, max_len_a
        except Exception as e:
            raise CustomException(e,sys)
        
    def model_building(self, vocab_size, embedding_dim, hidden_units, maxlen_questions, maxlen_answers):
        if os.path.exists('artifacts/chatbot_model.pkl'):
            # Load existing model
            chatbot_model = load_model('artifacts/chatbot_model.pkl')
        else:
            chatbot_model = model(vocab_size, embedding_dim, hidden_units, maxlen_questions, maxlen_answers)
        return chatbot_model
    
    def model_training(self, model, encoder_input_data, decoder_input_data, decoder_output_data, batch_size, epochs, validation_split):
        try:
            modelbuild = ChatbotTrainer()
            modelbuild.train_model(model, encoder_input_data, decoder_input_data, decoder_output_data, batch_size, epochs, validation_split)
        except Exception as e:
            raise CustomException(e,sys)
                
       
    def start_trainig(self,batch_size=64,epochs=50,validation_split=0.18,embedding_dim=256,hidden_units=512):
        try:
            ques,ans = self.start_data_ingestion()

            token,vocab_size = self.start_data_preprocessing(ques,ans)
            logging.info(f"VOCAB SIZE used for training: {vocab_size}")
            encoder_input, decoder_input, decoder_output, max_len_q, max_len_a=self.start_data_transformation(token,ques,ans,vocab_size)
            print(max_len_q,max_len_a)
            model = self.model_building(vocab_size, embedding_dim, hidden_units, max_len_q, max_len_a)
            self.model_training(model, encoder_input, decoder_input, decoder_output, batch_size, epochs, validation_split)
            
        except Exception as e:
            raise CustomException(e,sys)
        

