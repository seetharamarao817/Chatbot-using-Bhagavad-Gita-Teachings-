from src.logging.logger import logging
from src.exceptions.exception import CustomException
import sys
import numpy as np
from tensorflow.keras import preprocessing, utils

class DataTransformer:
    def __init__(self, tokenizer,questions,answers,vocab_size):
        self.tokenizer = tokenizer
        self.questions = questions
        self.answers = answers
        self.vocab_size = vocab_size
        

    def transform_data(self):
        try:
            # Encoder Input Data
            tokenized_questions = self.tokenizer.texts_to_sequences(self.questions )
            maxlen_questions = max( [len(x) for x in tokenized_questions ] )
            padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions, maxlen = maxlen_questions, padding = 'post')
            encoder_input_data = np.array(padded_questions)
            logging.info(encoder_input_data.shape, maxlen_questions)

            # Decoder Input Data
            tokenized_answers = self.tokenizer.texts_to_sequences(self.answers)
            maxlen_answers = max([len(x) for x in tokenized_answers])
            padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers,padding='post')
            decoder_input_data = np.array(padded_answers)
            logging.info(decoder_input_data.shape, maxlen_answers)


            # Decoder Output Data
            for i in range(len(tokenized_answers)):
                tokenized_answers[i] = tokenized_answers[i][1:]
            padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
            onehot_answers = utils.to_categorical(padded_answers, self.vocab_size)
            decoder_output_data = np.array(onehot_answers)
            logging.info(decoder_output_data.shape)

            return encoder_input_data, decoder_input_data, decoder_output_data, maxlen_questions, maxlen_answers

        except Exception as e:
            logging.error("Error occurred in data transformation part: {}".format(e))
            raise CustomException(e, sys)