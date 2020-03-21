from flask import Flask, render_template, request, redirect, url_for
import pickle
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Return homepage."""
    return render_template('index.html')

@app.route('/detector')
def detector():
    """Return bias detector page."""
    with open('env/bias_model.pkl', 'rb') as file:
        bias_model = pickle.load(file)

    bias = bias_model.predict(['I love Donald Trump, building the border is the best thing we could have done. I also love capitalism too.'])

    return render_template('detector.html', bias_prediction=bias)

@app.route('/about')
def about():
    """Return about page."""
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
