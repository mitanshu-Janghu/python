import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "hours_studied": [1, 2, 3, 4, 5],
    "marks": [35, 40, 50, 60, 65]
}

df = pd.DataFrame(data)

X = df[["hours_studied"]]   
y = df["marks"]             

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Prediction for 6 hours:", model.predict([[6]])[0])