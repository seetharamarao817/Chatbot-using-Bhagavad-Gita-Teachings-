import os,sys
from src.logging.logger import logging
from src.exceptions.exception import CustomException
from src.utils.utils import loading_model
from src.components.model_prediction import prediction

class PredictionPipeline:
    def loaded_model(self,filepath):
        try:
            model = loading_model(filepath)
            return model
        except Exception as e:
            raise CustomException(e,sys)
            
    def generateresponse(self,input_text,modelpath,tokenizerpath,maxlen_questions=38,maxlen_answers=168):
        model = self.loaded_model(modelpath)
        Inference = prediction(model,tokenizerpath)
        response = Inference.generate_response(input_text,maxlen_questions,maxlen_answers)
        return response
    
    def interactive_chat(self,modelpath,tokenizerpath):
        try:
            
            logging.info("Starting interactive chatbot... (Type 'exit' to end)")
 
            while True:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    logging.info("Exiting interactive chatbot.")
                    break

                response = self.generateresponse(input_text=user_input,modelpath=modelpath,tokenizerpath=tokenizerpath)
                print("Chatbot:", response)

        except Exception as e:
            logging.error("Error occurred in interactive chat.")
            raise CustomException(e, sys)
        

if __name__ =="__main__":
    predict = PredictionPipeline()
    predict.interactive_chat(modelpath="/workspaces/Chatbot-using-Bhagavad-Gita-Teachings-/artifacts/chatbot_model.pkl",tokenizerpath="/workspaces/Chatbot-using-Bhagavad-Gita-Teachings-/artifacts/tokenizer.json")