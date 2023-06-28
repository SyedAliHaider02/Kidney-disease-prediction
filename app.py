from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle


model = pickle.load(open('kidney1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('front.html')


@app.route('/predict', methods=['GET', 'POST'])
def home():
    data1 = request.form['id']
    data2 = request.form['age']
    data3 = request.form['bp']
    data4 = request.form['sg']
    data5 = request.form['al']
    data6 = request.form['su']
    data7 = request.form['rbc']
    data8 = request.form['pc']
    data9 = request.form['pcc']
    data10 = request.form['ba']
    data11 = request.form['bgr']
    data12 = request.form['bu']
    data13 = request.form['sc']
    data14 = request.form['sd']
    data15 = request.form['pot']
    data16 = request.form['hemo']
    data17 = request.form['pcv']
    data18 = request.form['wc']
    data19 = request.form['rc']
    data20 = request.form['htn']
    data21 = request.form['dm']
    data22 = request.form['cad']
    data23 = request.form['appet']
    data24 = request.form['pe']
    data25 = request.form['ane']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13,data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25]])
    pred = model.predict(arr)
    return render_template('home.html', data=pred)

    

if __name__=="__main__":
    app.run(debug=True)