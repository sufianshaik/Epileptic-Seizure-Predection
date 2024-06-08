
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import sqlite3
import pickle


app = Flask(__name__)

mo     

file = open("models/model.pkl","rb")
svm_model = pickle.load(file)
file.close()

file = open("models/scaler.pkl","rb")
scaler = pickle.load(file)
file.close()
 


@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        labels = ['Interictal State (Normal)', 'Preictal State (Seizure)']
        file = request.files['files']
        if file.filename == '':
            return render_template('index.html', message='No file selected')

        if file:
            
            dataset = pd.read_csv(file)
    
            print(dataset)
            dataset = dataset.values
            testData = scaler.transform(dataset)
            testData = np.reshape(testData, (testData.shape[0], 20, 20, 3))
            feature_extractor = Model(model.inputs, model.layers[-2].output)
            model_features = feature_extractor.predict(testData) 
            predict_svm = svm_model.predict(model_features)
            print(predict_svm)
            results = []
            for i in range(len(predict_svm)):
                result = {
                    "Test Data": str(dataset[i]),  # Access NumPy array directly
                    "Predicted Label": labels[int(predict_svm[i])]
                }
                results.append(result)

            # Return the appropriate response
            return render_template('after.html', results=results)

    return render_template('index.html')
    
    
@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/notebook')
def notebook():
	return render_template('NOtebook.html')


if __name__ == '__main__':
    app.run(debug=False)