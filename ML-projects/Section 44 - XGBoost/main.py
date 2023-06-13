import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = [0 if num == 2 else 1 for num in y_train]
y_test = [0 if num == 2 else 1 for num in y_test]
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred=y_pred, y_true=y_test)
print(cm)
ac_sc = accuracy_score(y_pred=y_pred, y_true=y_test)
print(ac_sc)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# parameters = [{"C": [0.25, 0.5, 0.75, 1], "kernel": ['linear']},
#               {"C": [0.25, 0.5, 0.75, 1], "kernel": ['rbf'], "gamma": [x/10 for x in range(1, 10)]}]
#
# grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
#
# grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
#
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)
