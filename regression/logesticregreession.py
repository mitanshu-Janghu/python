import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7],
    "attendance": [60, 65, 70, 75, 80, 85, 90],
    "pass": [0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["hours_studied", "attendance"]]  
y = df["pass"]                           

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Probability of passing:", model.predict_proba([[6, 85]])[0])