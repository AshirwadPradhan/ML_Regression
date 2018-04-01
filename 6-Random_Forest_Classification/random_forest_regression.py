import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# importing the dataset
dataset = pd.read_csv('data\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# feature scaling
# sc_X = StandardScaler()
# X = sc_X.fit_transform(X)
# sc_y = StandardScaler()
# y = sc_y.fit_transform(y)

# fitting SVR to the dataset
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X, y)

# predicting the slaray for positon 6.5
# inverse transform due to feature scaling
y_pred = regressor.predict(6.5)

# visualizing the Decision Tree result
# this is a non-linear as well as non-continuos regression model
# therefore this visualization will not work
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff: Random Forest')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# non-continuos regression model needs to be plotted in higher
# resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff: Random Forest')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
