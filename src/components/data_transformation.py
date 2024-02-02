from src.logging.logger import logging
from src.exceptions.exception import CustomException
import sys
import numpy as np
from tensorflow.keras import preprocessing, utils

class DataTransformer:
    def __init__(self, tokenizer, answers, vocab_size):
        self.tokenizer = tokenizer
        self.answers = answers
        self.vocab_size = vocab_size

    def transform_data(self):
        try:
            # Encoder Input Data
            tokenized_questions = self.tokenizer.texts_to_sequences(self.answers)
            maxlen_questions = max(len(x) for x in tokenized_questions)
            encoder_input_data = self.preprocess_data(tokenized_questions, maxlen_questions)
            logging.info(encoder_input_data.shape, maxlen_questions)

            # Decoder Input Data
            tokenized_answers = self.tokenizer.texts_to_sequences(self.answers)
            maxlen_answers = max(len(x) for x in tokenized_answers)
            decoder_input_data = self.preprocess_data(tokenized_answers, maxlen_answers)
            logging.info(decoder_input_data.shape, maxlen_answers)

            # Decoder Output Data
            for i in range(len(tokenized_answers)):
                tokenized_answers[i] = tokenized_answers[i][1:]
            padded_answers = self.preprocess_data(tokenized_answers, maxlen_answers)
            onehot_answers = utils.to_categorical(padded_answers, self.vocab_size)
            decoder_output_data = np.array(onehot_answers)
            logging.info(decoder_output_data.shape)

            return encoder_input_data, decoder_input_data, decoder_output_data, maxlen_questions, maxlen_answers

        except Exception as e:
            logging.error("Error occurred in data transformation part: {}".format(e))
            raise CustomException(e, sys)

    @staticmethod
    def preprocess_data(tokenizer, data, maxlen):
        tokenized_data = tokenizer.texts_to_sequences(data)
        padded_data = preprocessing.sequence.pad_sequences(tokenized_data, maxlen=maxlen, padding='post')
        return np.array(padded_data)
        



# Example usage:
# transformer = DataTransformer(tokenizer, answers, vocab_size)
# encoder_input, decoder_input, decoder_output, max_len_q, max_len_a = transformer.transform_data()
