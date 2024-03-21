import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


path = ('C:/Users/mpran/Documents/Git/'
        'bullbeary_bots_ai/machine_learning/machine_learning_a_z/datasets/Position_Salaries.csv')
dataset = pd.read_csv(path)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Implement simple linear regression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Implement polynomial linear regression
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Plot linear regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary (USD)")
plt.show()

# Plot polynomial regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary (USD)")
plt.show()

# Smoother plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary (USD)")
plt.show()

# Predicting the results
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
