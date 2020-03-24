from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import joblib

app = Flask(__name__)

with open('bias_model.pkl', 'rb') as file:
    bias_model = joblib.load(file)

@app.route('/')
def index():
    """Return homepage."""
    return render_template('index.html')


@app.route('/detector', methods=['GET', 'POST'])
def detector():
    """Return bias detector page."""
    # Tried to speed up the process on Heruko specifcally with AWS.
    # Turned out to be slower
    # s3 = boto3.resource('s3',
    #  aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    #  aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    #
    # bias_model = BytesIO(s3.Bucket("bias-detector").Object("bias_model.pkl").get()['Body'].read())
    # with open(bias_model, 'rb') as file:
    #     content = pickle.load(file)

    if 'bias_text' in request.form:
        bias_text = request.form['bias_text']
        textbox = request.form['bias_text']
        bias = bias_model.predict([bias_text])
    else:
        bias = ''
        textbox = 'Your text here'


    return render_template('detector.html', bias_prediction=bias, textbox=textbox)

@app.route('/about')
def about():
    """Return about page."""
    return render_template('about.html')


if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
