import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
data = pd.read_csv("credit_data.csv")
features = data[["income", "age", "loan"]]
target = data["default"]

X = np.array(features).reshape(-1, 3)
y = np.array(target).reshape(-1, 1)

X = preprocessing.MinMaxScaler().fit_transform(X)
features_train, features_test, target_train, target_test = train_test_split(
    X, y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=20)
fitted_model = model.fit(features_train, target_train)
predictions = fitted_model.predict(features_test)
cross_validate_scores = []

for k in range(1, 100):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cross_validate_scores.append(scores.mean())
print('Optimal k value:', cross_validate_scores.index(
    max(cross_validate_scores)) + 1)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
