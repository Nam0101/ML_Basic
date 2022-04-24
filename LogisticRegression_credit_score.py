import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


credit_data = pd.read_csv("credit_data.csv")


features = credit_data[["income", "age", "loan"]]
target = credit_data["default"]

features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3)

model = LogisticRegression()
model.fit(features_train, target_train)

predictions = model.predict(features_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
