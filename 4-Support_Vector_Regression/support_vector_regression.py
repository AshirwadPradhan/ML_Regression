import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# importing the dataset
dataset = pd.read_csv('data\Position_Salaries.csv')
X = dataset.iloc[:, 1:2]
y = dataset.iloc[:, 2:3]

# SVR class does not have feature scaling
# feature scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# fitting SVR to the dataset
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# predicting the slaray for positon 6.5
# inverse transform due to feature scaling
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform
                    (np.array([6.5]).reshape(-1,1))))

# visualizing the SVR result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff: SVR')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
