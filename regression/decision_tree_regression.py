import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


path = ('C:/Users/mpran/Documents/Git/bullbeary_bots_ai/machine_learning'
        '/machine_learning_a_z/datasets/Position_Salaries.csv')
dataset = pd.read_csv(path)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# We don't need to scale any values for any kind of decision tree algorithms
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting the results
regressor.predict([[6.5]])

# Visualizing the results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Decision Tree Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary (USD)")
plt.show()
