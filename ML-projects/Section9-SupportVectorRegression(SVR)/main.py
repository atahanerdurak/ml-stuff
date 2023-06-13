import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np

# take data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)  # make it vertical to scale
print(X)
print(y)

# refactoring X and y
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

plt.figure(1)
# Now we cannot directly plot X and y because they are scaled ,so we apply inverse transform
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
# We apply inverse transform to X, and we predict the y values. However, we still need to inverse transform them too.
# Before that, we need to reshape to make it vertical.
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.figure(2)
# Smoother curve. We are getting the max and min values of X, but they are scaled. So we apply inverse transform first.
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
# now we reshape to make it vertical
X_grid = X_grid.reshape(-1, 1)
# Same thing we did in the first plot.
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
# No need to inverse transform X_grid because they are not scaled. However, in order to make the prediction, we first
# need to transform them. After the prediction, we should reshape them and apply inverse transform.
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (Support Vector Regression) (HD)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
