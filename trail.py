import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression,Perceptron
import seaborn as sb
import matplotlib.pyplot as plt
dataset=pd.read_csv("/Users/mitanshujanghu/trail.csv")
ss=StandardScaler()
kn=KNeighborsClassifier(n_neighbors=1)
x=dataset.iloc[:,:-1]
y=dataset["select"]
x=ss.fit_transform(x)
dt=DecisionTreeClassifier(max_depth=4)
lr=LogisticRegression()
lr1=LinearRegression()
pr=Perceptron()
min=dataset["money"].mean() - 3*dataset["money"].std()
max=dataset["money"].mean() + 3*dataset["money"].std()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=10)
dt.fit(x_train,y_train)
kn.fit(x_train,y_train)
# sb.scatterplot(x="money",y="cgpa",data=dataset,hue="select")
# plt.show()
pr.fit(x_train,y_train)
# lr.fit(x_train,y_train)
print(pr.score(x_test,y_test)*100)
print(pr.predict([[-100,12]]))