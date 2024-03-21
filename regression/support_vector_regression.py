import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


path = ('C:/Users/mpran/Documents/Git/machine_learning_templates/datasets/Position_Salaries.csv')
dataset = pd.read_csv(path)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Training the model
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting the results
prediction_scaled = regressor.predict(sc.transform([[6.5]]))
prediction = sc_y.inverse_transform(prediction_scaled.reshape(-1, 1))

# Visualizing the results
plt.scatter(sc.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title("Support Vector Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary (USD)")
plt.show()

