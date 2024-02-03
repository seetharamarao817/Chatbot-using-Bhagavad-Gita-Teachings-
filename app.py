from flask import Flask, render_template, request, jsonify
from pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response')
def get_response():
    user_message = request.args.get('user_message')
    response = prediction_pipeline.generateresponse(modelpath, tokenizerpath, user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    modelpath = 'your_model_folder/your_model_file.h5'
    tokenizerpath = 'your_model_folder/your_tokenizer_file.pickle'
    prediction_pipeline = PredictionPipeline()
    app.run(debug=True)
