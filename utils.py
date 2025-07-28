
import pandas as pd
import pickle

def load_data():
    return pd.read_csv("data/employee_data.csv")

def preprocess(df):
    df['OverTime'] = df['OverTime'].map({"Yes": 1, "No": 0})
    return df[['Age', 'MonthlyIncome', 'JobSatisfaction', 'OverTime']]

def make_prediction(df):
    model = pickle.load(open("model.pkl", "rb"))
    df = preprocess(df)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return pred, prob
