from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import gzip

app = Flask(__name__)

@app.route('/')
def index():
    """Return homepage."""
    return render_template('index.html')

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    """Return bias detector page."""

    if 'bias_text' in request.form:
        with gzip.open('bias_model.pkl', 'rb') as file:
            bias_model = pickle.load(file)
        bias_text = request.form['bias_text']
        textbox = request.form['bias_text']
        bias = bias_model.predict([bias_text])
    else:
        bias = ' '
        textbox = 'Your text here'

    return render_template('detector.html', bias_prediction=bias, textbox=textbox)

@app.route('/about')
def about():
    """Return about page."""
    return render_template('about.html')


if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
