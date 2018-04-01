import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# import the data
dataset = pd.read_csv('data\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# train and test data split
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                        random_state=0, test_size=0.2)
# here we dont split bacause of lack of enough data

# fitting linear regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# fitting polynomial regression to the dataset
# creates new features x1^2, x1^3...,x1^n
# here we are only going to x1^2
# here also the column of ones is also created for x0
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)
# creating 2nd linear model
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# predicted values
y_pred1 = lin_reg.predict(X)

# we actually want the transformed values to X which is
# X_poly and not X
y_pred2 = lin_reg2.predict(poly_features.fit_transform(X))

# visualizing the Linear Regression model
plt.scatter(X, y, color='red')
plt.plot(X, y_pred1)
plt.title('Truth or Bluff! : Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualizing the Polynomial Regression model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
y_grid_pred = lin_reg2.predict(poly_features.fit_transform(X_grid))
plt.scatter(X, y, color='red')
plt.plot(X_grid, y_grid_pred, color='blue')
plt.title('Truth or Bluff! : Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# predicting the value of the Employee whose salary was
# 160k
# using SLR
# predicted value of 6.5 level
slr_truth = lin_reg.predict(6.5)
# using PLR
plr_truth = lin_reg2.predict(poly_features.fit_transform(6.5))
# we get 158k hence he was telling the truth
# therefore the employee is honest!!
