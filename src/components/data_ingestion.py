import os
import yaml
import json
from src.logging.logger import logging
from src.exceptions.exception import CustomException



class DataIngestor:
    def __init__(self):
        self.questions = []
        self.answers = []

    def add_data_to_dataframe(self, dictionary):
        try:
            if "question" in dictionary and "answer" in dictionary:
                self.add_data_to_lists(dictionary["question"], ' start ' + dictionary["answer"] + ' end ')
            else:
                self.add_data_to_lists(f"explain chapter number : {dictionary['chapter_number']} {dictionary['translation']} or {dictionary['meaning']['en']}",
                                       ' start ' + dictionary['summary']['en'] + ' end ')
            logging.info("Data added to the dataframe successfully.")
        except Exception as e:
            logging.error("Error occurred while adding data to the dataframe.")
            raise CustomException(e)

    def read_yaml(self, file_path):
        try:
            with open(file_path, 'rb') as stream:
                docs = yaml.safe_load(stream)
            return docs['conversations']
        except Exception as e:
            logging.error(f"Error reading YAML file: {file_path}")
            raise CustomException(e)

    def read_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data_list = json.loads(file.read())
            return data_list
        except Exception as e:
            logging.error(f"Error reading JSON file: {file_path}")
            raise CustomException(e)

    def process_conversations(self, conversations):
        processed_questions = []
        processed_answers = []

        for con in conversations:
            if len(con) > 2:
                processed_questions.append(con[0])
                replies = con[1:]
                ans = ' '.join(rep for rep in replies)
                processed_answers.append(f' start {ans} end ')
            elif len(con) > 1:
                processed_questions.append(con[0])
                processed_answers.append(f' start {con[1]} end ')

        return processed_questions, processed_answers

    def add_data_to_lists(self, question, answer):
        self.questions.append(question)
        self.answers.append(answer)

    def data_ingestion(self):
        try:
            ls = []
            for dirname, _, filenames in os.walk('/kaggle/input'):
                if dirname == '/kaggle/input/bhagavad-gita-api-database/chapter':
                    for filename in filenames:
                        ls.append(os.path.join(dirname, filename))

            files_list = [
                "/kaggle/input/chatterbotenglish/ai.yml",
                "/kaggle/input/chatterbotenglish/greetings.yml"
            ]

            for filepath in files_list:
                conversations = self.read_yaml(filepath)
                processed_questions, processed_answers = self.process_conversations(conversations)
                self.questions.extend(processed_questions)
                self.answers.extend(processed_answers)

            file_paths = [
                "/kaggle/input/gita-chatbot/Ethical_scenarios",
                "/kaggle/input/gita-chatbot/Leadership_development",
                "/kaggle/input/gita-chatbot/greetings.txt"
            ]

            for file_path in file_paths:
                data_list = self.read_json(file_path)
                for dictionary in data_list:
                    self.add_data_to_dataframe(dictionary)

            for file_path in ls:
                data_list = self.read_json(file_path)
                self.add_data_to_dataframe(data_list)

            logging.info("Data ingestion completed successfully.")
            return processed_questions,processed_answers
        
        except Exception as ce:
            
            raise CustomException(ce)
    

