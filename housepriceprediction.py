import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("housing1.csv")
del df["age"]
x = df.iloc[:,:-1]
y = df["price"]
X_train, X_test, y_train, y_test=train_test_split(
    x,y, test_size=0.3, random_state=42
)

model=LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual prices:", list(y_test))
print("Predicted prices:", y_pred)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))