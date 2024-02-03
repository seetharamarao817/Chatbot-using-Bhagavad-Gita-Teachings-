from src.logging.logger import logging
from src.exceptions.exception import CustomException
from src.utils.utils import load_tokenizer_from_json
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence as preprocessing



class prediction:
    def __init__(self,model,tokenizerpath):
        self.tokenizer = load_tokenizer_from_json(tokenizerpath)
        self.reverse_word_index = {index: word for word, index in self.tokenizer.word_index.items()}
        self.model = model
        pass

    def generate_response(self,input_text,maxlen_questions,maxlen_answers):
        try:
            
            # Tokenize the input text
            input_seq = self.tokenizer.texts_to_sequences([input_text])
            input_seq = preprocessing.pad_sequences(input_seq, maxlen=maxlen_questions, padding='post')
            input_seq = self.model.layers[1](input_seq)

            # Encode the input sequence
            encoder_output, forward_h, forward_c, backward_h, backward_c = self.model.layers[3](input_seq)
            encoder_states = [self.model.layers[5]([forward_h, backward_h]), self.model.layers[6]([forward_c, backward_c])]

            # Initialize the target sequence with a start token
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = self.tokenizer.word_index['start']

            stop_condition = False
            generated_response = ''

            while not stop_condition:
                # Predict the next word
                target_seq = self.model.layers[2](target_seq)
                target_seq = self.model.layers[4](target_seq)
                output_tokens, h, c = self.model.layers[7](inputs=target_seq, initial_state=encoder_states)
                attention_weights = self.model.layers[8]([output_tokens, encoder_output])
                output_tokens = self.model.layers[9]([output_tokens, attention_weights])
                output_tokens = self.model.layers[10](output_tokens)
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_word = self.reverse_word_index.get(sampled_token_index, 'unknown')


                # Exit condition: either hitting max length or finding the stop token
                if sampled_word == 'end' or len(generated_response.split()) > maxlen_answers:
                    stop_condition = True
                else:
                    generated_response += sampled_word + ' '

                # Update the target sequence for the next iteration
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index

                # Update states
                encoder_states = [h, c]

            return generated_response.strip()

        except Exception as e:
            logging.error("Error occurred in generating response.")
            raise CustomException(e, sys)