import sys
from tensorflow.keras import preprocessing
import numpy as np
import re
from src.exceptions.exception import CustomException
from src.logging.logger import logging
from gensim.models import Word2Vec

class DataPreprocessor:
    def __init__(self, questions, answers):
        self.tokenizer = preprocessing.text.Tokenizer()
        self.questions = questions
        self.answers = answers
        self.maxlen = 0

    def _tokenize_sentences(self, sentences):
        tokens_list = []
        vocabulary = []

        for sentence in sentences:
            sentence = sentence.lower()
            sentence = re.sub('[^a-zA-Z]', ' ', sentence)
            tokens = sentence.split()
            vocabulary += tokens
            tokens_list.append(tokens)

        return tokens_list, vocabulary

    def _create_tokenizer(self):
        try:
            self.tokenizer.fit_on_texts(self.questions + self.answers)
            vocab_size = len(self.tokenizer.word_index) + 1
            logging.info(f'VOCAB SIZE: {vocab_size}')
        except Exception as e:
            logging.info("Error occurred in creating tokenizer")
            raise CustomException(e, sys)

    def _preprocess_data(self, data):
        try:
            tokenized_data = self.tokenizer.texts_to_sequences(data)
            padded_data = preprocessing.sequence.pad_sequences(tokenized_data, maxlen=self.maxlen, padding='post')
            return np.array(padded_data)
        except Exception as e:
            logging.info("Error occurred in data preprocessing")
            raise CustomException(e, sys)

    def preprocess(self):
        try:
            self._create_tokenizer()
            tokenized_questions, _ = self._tokenize_sentences(self.questions)
            tokenized_answers, _ = self._tokenize_sentences(self.answers)

            max_len_questions = max(len(sentence) for sentence in tokenized_questions)
            max_len_answers = max(len(sentence) for sentence in tokenized_answers)
            self.maxlen = max(max_len_questions, max_len_answers)

            preprocessed_questions = self._preprocess_data(tokenized_questions)
            preprocessed_answers = self._preprocess_data(tokenized_answers)

            return preprocessed_questions, preprocessed_answers
        except CustomException as ce:
            logging.error(f"CustomException: {ce}")
    
            raise CustomException(ce, sys)
