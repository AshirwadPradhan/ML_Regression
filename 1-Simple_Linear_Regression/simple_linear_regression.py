import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr

# import the dataset
dataset = pd.read_csv('data\Salary_Data.csv')
X = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]

# train and test split
X_train, X_test, y_train, y_test = tts(X, y, test_size=1/3,
                                       random_state=0)

# feature scaling
# not needed for SLR

# fitting SLR to the training set
regressor = lr()
regressor.fit(X_train.values.reshape(-1, 1), y_train)
# reshape is used bcoz the fit expects 2D array but we have ony one
# feature hence reshape(-1, 1) converts this.

# predicting the test set results
y_test_pred = regressor.predict(X_test.values.reshape(-1, 1))

# visualize the training set results
plt.scatter(X_train, y_train, color='red')
y_train_pred = regressor.predict(X_train.values.reshape(-1, 1))
plt.plot(X_train, y_train_pred, color='blue')
plt.title('Salary vs Experience:Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualize the test set results
plt.scatter(X_test, y_test, color='red')
y_train_pred = regressor.predict(X_train.values.reshape(-1, 1))
plt.plot(X_train, y_train_pred, color='blue')
plt.title('Salary vs Experience:Test Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
