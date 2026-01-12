import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
df=pd.read_csv("IRIS.csv")
df["species"] = df["species"].replace({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).infer_objects(copy=False)
inp=df.iloc[:,:-1]
ans=df["species"]
model=KNeighborsClassifier()
x_train,x_test,y_train,y_test=train_test_split(inp,ans)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("accuray" , accuracy_score(y_pred,y_test))