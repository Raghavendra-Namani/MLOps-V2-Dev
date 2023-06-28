import os
import subprocess
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

import mlflow

def test_evaluate_model():
    
    test_data = "/tmp/test"
    model_input = "/tmp/model"
    evaluation_output = "/tmp/evaluate"
    model_name = "taxi-model"
    runner = "LocalRunner"

    os.makedirs(test_data, exist_ok = True)
    os.makedirs(model_input, exist_ok = True)
    os.makedirs(evaluation_output, exist_ok = True)


    data = {
        'PatientID': [1354778,1147438,1640031,1883350,1424119,1619297,1660149,1458769,1201647,
        1403912,1943830,1824483,1848869,1669231,1683688,1738587,1884264,1485251,1536832,1438701],
        'Pregnancies': [0,8,7,9,1,0,0,0,8,1,1,3,5,7,0,3,3,1,8,3],
        'PlasmaGlucose': [171,92,115,103,85,82,133,67,80,72,88,94,114,110,148,109,106,156,117,102],
        'DiastolicBloodPressure': [80,93,47,78,59,92,47,87,95,31,86,96,101,82,58,77,64,53,39,100],
        'TricepsThickness': [34,47,52,25,27,9,19,43,33,40,11,31,43,16,11,46,25,15,32,25],
        'SerumInsulin': [23,36,35,304,35,253,227,36,24,42,58,36,70,44,179,61,51,226,164,289],
        'BMI': [43.50972593,21.24057571,41.51152348,29.58219193,42.60453585,19.72416021,21.94135672,18.2777226,26.62492885,36.88957571,43.22504089,21.29447943,36.49531966,36.08929341,39.19207553,19.84731197,29.0445728,29.78619164,21.23099598,42.18572029],
        'DiabetesPedigree': [1.213191354,0.158364981,0.079018568,1.282869847,0.549541871,0.103424498,0.174159779,0.23616494,0.443947388,0.103943637,0.230284623,0.259020482,0.079190164,0.281276159,0.160829008,0.204345272,0.589188017,0.203823525,0.089362745,0.175592826],
        'Age': [21,23,23,43,22,26,21,26,53,26,22,23,38,25,45,21,42,41,25,43],
        'Diabetic': [0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,1,1,0,1]
    }


    # Save the data
    df = pd.DataFrame(data)
    df.to_parquet(os.path.join(test_data, "test.parquet"))

    # Split the data into inputs and outputs
    y_test = df["Diabetic"]
    X_test = df.drop(['PatientID','Diabetic'],axis=1)

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(random_state=0)
    model.fit(X_test, y_test)

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=model_input)


    cmd = f"python data-science/src/evaluate/evaluate.py --model_name={model_name} --model_input={model_input} --test_data={test_data} --evaluation_output={evaluation_output} --runner={runner}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate() 
    result = str(out).split('\\n')
    for lin in result:
        if not lin.startswith('#'):
            print(lin)
    
    assert os.path.exists(os.path.join(evaluation_output, "predictions.csv"))
    assert os.path.exists(os.path.join(evaluation_output, "score.txt"))
    
    print("Train Model Unit Test Completed")

if __name__ == "__main__":
    test_evaluate_model()
