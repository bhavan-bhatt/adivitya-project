from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("heart_disease_detector.pkl", "rb"))



@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")




@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        age = request.form["age"]
        sex = request.form["sex"]
        #if (gender == '1'):
        #    sex = 1
        #elif (gender == '0'):
        #    sex = 0
        cp = request.form["cp"]
        trestbps = request.form["trestbps"]
        chol = request.form["chol"]
        fbs = request.form["fbs"]
        restecg = request.form["restecg"]
        thalach = request.form["thalach"]
        exang = request.form["exang"]
        oldpeak = request.form["oldpeak"]
        slope = request.form["slope"]
        ca = request.form["ca"]
        thal = request.form["thal"]

        prediction=model.predict([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        output=prediction

        if (output==1):
             return render_template("home.html",prediction_text="You have Heart Disease".format(output))
        elif(output==0):
            return render_template("home.html",prediction_text="You have Heart Disease".format(output))

       


    return render_template("home.html")




if __name__ == "__main__":
    app.run(debug=True)
