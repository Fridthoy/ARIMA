from flask import Flask, render_template, request, url_for
import pandas as pd
from webBackend import *

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def hello():
    return render_template('index.html')



@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':

        file = request.files['upload-file']
        print(file)
        df = testData(file)
        print(df)
        step= 5
        print(checkStationarity(df))
        findOrder(df)
        pcorr = create_autocorr_p(df)
        qcorr = create_autocorr_q(df)
        webname = websiteArima(df, step)
        predName = createPredictions(df, step)

        return render_template('return.html',
            graph= webname, pred= predName, pcorr = pcorr, qcorr = qcorr)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)