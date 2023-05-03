import pandas as pd
from pycaret.classification import *

diabetes = pd.read_csv('data/diabetes.csv')

clf = setup(data=diabetes, target='Outcome')
best_model = compare_models()
model = create_model(best_model)
new_data = pd.DataFrame({
    'Pregnancies': [6, 1, 8],
    'Glucose': [148, 85, 183],
    'BloodPressure': [72, 66, 64],
    'SkinThickness': [35, 29, 0],
    'Insulin': [0, 0, 0],
    'BMI': [33.6, 26.6, 23.3],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672],
    'Age': [50, 31, 32]
})
predictions = predict_model(model, data=new_data)
