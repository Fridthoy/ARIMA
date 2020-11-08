from flask import Flask, render_template, request, url_for
import pandas as pd
from main import *

app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def hello():
    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.form['upload-file']
        df = testData(file)
        websiteArima(df)
        print(df)
        print(df.dtypes)
        return render_template('return.html')

if __name__ == '__main__':
    app.run(debug=True)