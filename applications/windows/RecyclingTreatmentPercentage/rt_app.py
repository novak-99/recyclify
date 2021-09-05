import numpy as np 
import pandas as pd
import sklearn
import pickle

FILE_NAME = 'data.txt'

with open('assets/rt_model','rb') as file:
  model = pickle.load(file)

with open('assets/rt_scaler','rb') as file:
  sc = pickle.load(file)

df = pd.read_csv(FILE_NAME)

X_inference = df.to_numpy()
X_inference = sc.transform(X_inference)

y_hat = model.predict(X_inference)
print("Predicting...")
print(y_hat)

pred_file = open("predictions.txt","w")
pred_file.write("Predicted a value of: " + str(y_hat[0]))
pred_file.close()