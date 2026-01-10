import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


data = {
    "hours_studied": [1,2,3,4,5,6,7,8,9,10,11],
    "attendance": [60,65,70,75,80,85,90,95,90,88,68],
    "pass": [0,0,0,1,1,1,1,1,1,0,1]
}

df = pd.DataFrame(data)

X = df[["hours_studied", "attendance"]]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = SVC(
    kernel="linear",   
    C=1.0
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Prediction:", model.predict([[6, 85]]))