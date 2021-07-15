from flask import Flask,render_template,url_for,request
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd



app = Flask(__name__)
model = pickle.load(open("NB_spam_model.pkl", "rb"))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
	if request.method == 'POST':
        
        messages = request.form['messages']
		data = [messages]
		vect = cv.transform(data).toarray()
		my_prediction = spam_detect_classifier.predict(vect)
    		
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)