from src.logging.logger import logging
from src.exceptions.exception import CustomException
import sys
from tensorflow.keras.optimizers import Adam

class ChatbotTrainer:
    def __init__(self):
        pass

    def train_model(self, model, encoder_input_data, decoder_input_data, decoder_output_data, batch_size, epochs, validation_split):
        try:
            logging.info("Model training started")
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
                      batch_size=batch_size, epochs=epochs, validation_split=validation_split)

            # Save the trained model
            model.save('chatbot_model.pkl')
            logging.info("Model training completed successfully.")

        except Exception as e:
            logging.error("Error occurred in model training part: {}".format(str(e)))
            raise CustomException(e, sys)

