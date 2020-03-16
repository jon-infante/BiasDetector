from flask import Flask, render_template, request, redirect, url_for
import requests
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Return homepage."""
    return render_template('index.html', msg='Home')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))
