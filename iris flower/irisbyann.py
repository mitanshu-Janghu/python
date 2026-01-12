import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("IRIS.csv")
df["species"].replace({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
},inplace=True)
x=df.iloc[:,:-1]
y=df["species"]
model=MLPClassifier(
    hidden_layer_sizes=(5, 12),  
    solver="adam",
    max_iter=1000,
    random_state=7
)
x_train,x_test,y_train,y_test=train_test_split(x,y)
model.fit(x,y)
y_pred=model.predict(x_test)

print("accuracy",accuracy_score(y_pred,y_test))
