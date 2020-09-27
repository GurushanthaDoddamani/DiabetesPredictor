import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('DiabetesPredictor.pkl', 'rb'))
#ohe = pickle.load(open('StateEncoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    pregnancies = int(request.form['pregnancies'])
    glucose = int(request.form['glucose'])
    bloodPressure = int(request.form['bloodPressure'])
    skinThickness = int(request.form['skinThickness'])
    insulin = int(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetesPedigreeFunction = float(request.form['diabetesPedigreeFunction'])
    age = int(request.form['age'])
    feature1Array = np.array([[pregnancies,glucose,bloodPressure,skinThickness,insulin,bmi,
                           diabetesPedigreeFunction,age]])
    outcome = model.predict(feature1Array)
    return render_template('index.html', prediction_text='The person is {}'.format(round(outcome[0])))



if __name__ == "__main__":
    app.run(debug=True)
    '''
    For rendering results on HTML GUI
    '''









   # rdSpend = float(request.form['rdSpend'])
   # admSpend = float(request.form['admSpend'])
   # markSpend = float(request.form['markSpend'])
   # state = request.form['state']
    #stateEncoded = ohe.transform(np.array([[state]]))
    #finalFeatures = np.concatenate((stateEncoded,np.array([[rdSpend,admSpend,markSpend]])) , axis = 1)
    #prediction = model.predict(finalFeatures)

    




    

