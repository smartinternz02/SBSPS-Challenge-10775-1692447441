
import requests
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Replace <your API key> with your actual IBM Cloud API Key
API_KEY = "ieh0Rg1i0ylPfwaAhRuHaUTfT4-YZLukbKVzTOVO-SWc"

# Define the URL of your deployed model
MODEL_URL = 'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/ae274275-95f2-4078-b951-5b959edc8b1c/predictions?version=2021-05-01'

#preprocessing for regression
scaler=pickle.load(open('preprocess.pkl','rb'))


@app.route('/')
def helloworld():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    a = request.form["gender"]
    b = request.form["ssc"]
    c = request.form["hse"]
    d = request.form["hsep"] 
    e = request.form["dp"]
    f = request.form["df"]
    g = request.form["we"]
    h = request.form["etp"]
    i = request.form["mbasp"]
    j = request.form["mbap"]

    # Preprocess the input data
    # 1. Convert categorical variables to numerical (if needed)
    a = 0 if a == 'f' else 1

    # 2. Encode 'hsep' and 'df' using one-hot encoding or label encoding
    if d == "comm":
        d = 1
    elif d == "scie":
        d = 2
    else:
        d = 0

    if f == "commMang":
        f = 0
    elif f == "sciTech":
        f = 2
    else:
        f = 1

    if i == "mktHr":
        i = 1
    else:
        i = 0

    print("aaa")
    # 4. Combine all the features into the final input vector
    t = [[int(a), float(b), float(c), int(d), float(e), int(f), float(g), float(h), int(i), float(j)]]
    #t=[[1,34,34,1,34,2,23,23,1,34]]

    # Get the access token
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    # Set the request headers
    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # Construct the payload for scoring
    payload_scoring = {
        "input_data": [{
            "fields": ['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t','workex', 'etest_p', 'specialisation', 'mba_p'],
            "values": t
        }]
    }

    # Send the prediction request to the model
    response_scoring = requests.post(MODEL_URL, json=payload_scoring, headers=header)

    # Process the prediction response
    output = response_scoring.json()
    
# Check if the prediction is "Placed" or "Not Placed"
    if 'predictions' in output and len(output['predictions']) > 0:
        prediction = output['predictions'][0]
        p=prediction['values'][0][0]

        if p == 0:
            verdict = "Not Placed"
            message = "Need to work Hard!!"
        else:
            verdict = "Placed"
            models_1=pickle.load(open('model.pkl', 'rb'))
            x = [[int(a), int(d), int(f), float(h), float(j)]]
            scaled_t = scaler.transform(x)
            salary = models_1.predict(scaled_t)
            message = f"The verdict is: {verdict} and salary is {np.round(salary[0])}"
    else:
    # Handle the case where there are no predictions or the format is unexpected
        verdict = "Prediction Error"
        message = "There was an error with the prediction response."

# Render the HTML template with the prediction results
    return render_template('index.html', y=f"The verdict is: {verdict} and {message}")


if __name__ == '__main__':
    app.run(debug=True)                                                                                                                                                                              







    
        