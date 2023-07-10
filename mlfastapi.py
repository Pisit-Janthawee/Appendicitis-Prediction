import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
import joblib
import os

from configuration.Pipeline import Pipeline


app = FastAPI(debug=True)

# Step
# 1. cd -> Main Directory of this file 
# 2. New Terminal 
# 3. uvicorn mlfastapi:app --reload

@app.get('/')
def home():
    return {'text': 'Appendicitis Prediction'}


@app.get('/predict')
def predict_disease(age, sex, height, body_weight, body_temperature, alcohol, smoking,
                    dysuria, anorexia, nausea_vomit, rebound, peritonitis, rlq, cough,
                    wbc, rbc, leukocytes):

    input_df = pd.DataFrame({'age': int(age),
                             'sex': sex,
                             'height': float(height),
                             'body_weight': float(body_weight),
                             'body_temperature': float(body_temperature),
                             'alcohol': alcohol,
                             'smoking': smoking,
                             'Dysuria': dysuria,
                             'Anorexia': anorexia,
                             'Nausea/vomiting': nausea_vomit,
                             'Rebound tenderness': rebound,
                             'Peritonitis/abdominal guarding': peritonitis,
                             'Tenderness in right lower quadrant': rlq,
                             'Cough tenderness': cough,
                             'WBC': float(wbc),
                             'RBC': float(rbc),
                             'Leukocytes': float(leukocytes),
                             'Alvarado Score (AS)': None,
                             'Pediatric appendicitis score (PAS)': None,
                             }, index=[0]).applymap(lambda x: x.lower() if isinstance(x, str) else x)
    # Data Preprocessing
    pipe = Pipeline()
    df_prep = pipe.preprocessing(input_df)
    selected_cols = ['age',
                     'sex',
                     'height',
                     'body_weight',
                     'body_temperature',
                     'alcohol',
                     'smoking',
                     'Dysuria',
                     'Anorexia',
                     'Nausea/vomiting',
                     'Rebound tenderness',
                     'Peritonitis/abdominal guarding',
                     'Tenderness in right lower quadrant',
                     'Cough tenderness',
                     'WBC',
                     'RBC',
                     'Leukocytes',
                     'Alvarado Score (AS)',
                     'Pediatric appendicitis score (PAS)']

    df_prep = df_prep[selected_cols]

    # Load the trained model
    path = 'D:\\Repo\\ML\\Classification\\Appendicitis\\model\\feature_importance\\grid_search\\grid_search_Logistic_Regression.pkl'
    model = joblib.load(path)


    # Make prediction
    pred = model.best_estimator_.predict_proba(df_prep)[0]


    output = {'No Disease': pred[0].round(3), 'Disease': pred[1].round(3)}
    return output


if __name__ == '__main__':
    uvicorn.run(app)
