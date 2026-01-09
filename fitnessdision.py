import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv(
    "fitness_utf8.csv",
    sep=",",
    engine="python",
    on_bad_lines="skip"
)
dt=DecisionTreeClassifier()
x=df.iloc[:,:-1]
y=df["attended"]
x_train,x_test,y_train,y_test=train_test_split(x,y)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))