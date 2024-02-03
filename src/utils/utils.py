import json
from keras.preprocessing.text import tokenizer_from_json

from tensorflow.keras.models import load_model

def loading_model(filepath):
    model = load_model(filepath)
    return model 

def load_tokenizer_from_json(file_path):
    """
    Loads a tokenizer from a JSON file.

    Args:
        file_path: The file path of the JSON file containing the tokenizer configuration.

    Returns:
        A Keras Tokenizer instance.
    """
    with open(file_path, 'r') as json_file:
        json_string = json_file.read()

    tokenizer = tokenizer_from_json(json_string)
    return tokenizer

def save_tokenizer_to_json(tokenizer, file_path, **kwargs):
        json_string = tokenizer.to_json(**kwargs)
        with open(file_path, 'w') as json_file:
            json_file.write(json_string)