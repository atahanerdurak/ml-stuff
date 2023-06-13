import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


print("1. Logistic Regression")
print("2. K-Nearest Neighbor")
print("3. Support Vector Machine")
print("4. Kernel SVM")
print("5. Naive Bayes")
print("6. Decision Tree")
print("7. Random Forest")
selection = int(input("Type the number for classification model: "))

if selection == 1:
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(cm)
    print(acc_score)

elif selection == 2:
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(cm)
    print(acc_score)

elif selection == 3:
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(cm)
    print(acc_score)

elif selection == 4:
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(cm)
    print(acc_score)

elif selection == 5:
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(cm)
    print(acc_score)
elif selection == 6:
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(cm)
    print(acc_score)
elif selection == 7:
    classifier = RandomForestClassifier(criterion='entropy', n_estimators=20)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    print(cm)
    print(acc_score)
else:
    raise ValueError("Invalid input. Please enter a valid number.")
