from flask import Flask, render_template, request, jsonify
from pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)
prediction_pipeline = PredictionPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    response = prediction_pipeline.generateresponse(input_text=user_message,modelpath="/workspaces/Chatbot-using-Bhagavad-Gita-Teachings-/artifacts/chatbot_model.pkl",tokenizerpath="/workspaces/Chatbot-using-Bhagavad-Gita-Teachings-/artifacts/tokenizer.json")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
