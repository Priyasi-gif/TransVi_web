from flask import Flask, render_template, request
from model_predict import predict_virus

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form['sequence']
    prediction = predict_virus(sequence)
    return render_template('index.html', prediction=prediction, input_sequence=sequence)

if __name__ == '__main__':
    app.run(debug=True)
