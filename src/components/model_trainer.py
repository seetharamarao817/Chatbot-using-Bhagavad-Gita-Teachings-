from src.logging.logger import logging
from src.exceptions.exception import CustomException
import sys
from tensorflow.keras.optimizers import Adam
from src.components.build_model import model

class ChatbotTrainer:
    def __init__(self):
        pass

   

    def train_model(self, model, encoder_input_data, decoder_input_data, decoder_output_data, batch_size, epochs, validation_split):
        try:
            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
                      batch_size=batch_size, epochs=epochs, validation_split=validation_split)

            # Save the trained model
            model.save('chatbot_model.h5')
            logging.info("Model training completed successfully.")

        except Exception as e:
            logging.error("Error occurred in model training part: {}".format(str(e)))
            raise CustomException(e, sys)

if __name__ == "__main__":
    vocab_size = 10000  # Replace with the actual vocabulary size
    embedding_dim = 256
    hidden_units = 512
    maxlen_questions = 30  # Replace with the actual maximum length of questions
    maxlen_answers = 30
   
    # Initialize the ChatbotTrainer class
    model = model(vocab_size, embedding_dim, hidden_units, maxlen_questions, maxlen_answers)

  
    trainer = ChatbotTrainer()
    # Call the function to train the model
    trainer.train_model(model, encoder_input_data, decoder_input_data, decoder_output_data, batch_size=64, epochs=15, validation_split=0.15)
