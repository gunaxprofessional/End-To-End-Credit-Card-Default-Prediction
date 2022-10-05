import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        Total_Trans_Ct = int(request.form['Total_Trans_Ct'])
        Total_Trans_Amt = int(request.form['Total_Trans_Amt'])
        Total_Revolving_Bal = int(request.form['Total_Revolving_Bal'])
        Months_Inactive_12_mon = int(request.form['Months_Inactive_12_mon'])
        Total_Ct_Chng_Q4_Q1 = float(request.form['Total_Ct_Chng_Q4_Q1'])
        Total_Relationship_Count = int(request.form['Total_Relationship_Count'])
        Avg_Utilization_Ratio = float(request.form['Avg_Utilization_Ratio'])
        Total_Amt_Chng_Q4_Q1 = float(request.form['Total_Amt_Chng_Q4_Q1'])
        Credit_Limit = int(request.form['Credit_Limit'])
        Contacts_Count_12_mon = int(request.form['Contacts_Count_12_mon'])

        features = [Total_Trans_Ct,Total_Trans_Amt,Total_Revolving_Bal,Months_Inactive_12_mon,Total_Ct_Chng_Q4_Q1,Total_Relationship_Count,Avg_Utilization_Ratio,Total_Amt_Chng_Q4_Q1,Credit_Limit,Contacts_Count_12_mon]
        features = np.array([features])
        # features = StandardScaler().fit_transform(features)
        prediction = model.predict(features)

        prediction_text01 = "This Customer is gonna default"
        if prediction == 0:
            prediction_text01 = "This Customer is not gonna default"
    
        return render_template('index.html',prediction_text = prediction_text01)


if __name__ == '__main__':
    app.run(debug=True)