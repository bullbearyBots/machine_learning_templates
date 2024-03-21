import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


path = 'C:/Users/mpran/Documents/Git/machine_learning_templates/datasets/Salary_Data.csv'
dataset = pd.read_csv(path)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set
y_pred = regressor.predict(X_test)

# Plot training set results
plt.scatter(X_train, y_train, color='red', label='Training set')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs. Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Plot test set results
plt.scatter(X_test, y_test, color='red', label='Testing set')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs. Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

print(regressor.predict([[12]]))
print(regressor.coef_)
print(regressor.intercept_)

# Equation would then be y = 9312.575 * Years of Experience + 26780.099
