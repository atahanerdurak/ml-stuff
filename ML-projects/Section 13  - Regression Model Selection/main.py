import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("1. Multiple Linear Regression")
print("2. Polynomial Regression")
print("3. Support Vector Regression")
print("4. Decision Tree Regression")
print("5. Random Forest Regression")
selection = int(input("Type the number for regression model: "))

if selection == 1:

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    print(f"Multiple Linear Regression R squared score: {r2_score(y_test, y_pred)}")

elif selection == 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    poly_regressor = PolynomialFeatures(degree=4)
    X_poly = poly_regressor.fit_transform(X_train)
    lin_regressor = LinearRegression()
    lin_regressor.fit(X_poly, y_train)
    y_pred = lin_regressor.predict(poly_regressor.fit_transform(X_test))

    print(f"Polynomial Regression R squared score: {r2_score(y_test, y_pred)}")

elif selection == 3:
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)

    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, y_train)

    y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)).reshape(-1, 1))

    print(f"Support Vector Regression R squared score: {r2_score(y_test, y_pred)}")

elif selection == 4:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(f"Support Vector Regression R squared score: {r2_score(y_test, y_pred)}")

elif selection == 5:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(f"Support Vector Regression R squared score: {r2_score(y_test, y_pred)}")