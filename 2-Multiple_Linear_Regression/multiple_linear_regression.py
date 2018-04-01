import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as stats

# importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# label encoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
# one hot encoding of column 3
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# avoiding the dummy variable trap
X = X[:, 1:]

# training and testing data split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=0)

# multiple linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set result
y_pred = regressor.predict(X_test)

# using backward elimination to make same model
# adding x0 = 1 in the model
X = np.append(values=X, arr=np.ones((50, 1)).astype(int), axis=1)
# x_opt contains only optimal features
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()
# removing the highest p value i.e. x2 > 0.05
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()
# removing next highest p value i.e. x1 > 0.05
X_opt = X[:,[0, 3, 4, 5]]
regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()
# removing next highest p value i.e x2 > 0.05
X_opt = X[:,[0, 3, 5]]
regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()
# removing next highest p value i.e x2 > 0.05
X_opt = X[:,[0, 3]]
regressor_ols = stats.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()
# so now we end up having only two relevant columns one of which we added
# is the row of ones and R&D spend. This to eliminate the other lesser
# relevant features and including only optimum features
