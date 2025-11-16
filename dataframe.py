import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,LogisticRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
import seaborn as sb 
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeClassifier
oe=OrdinalEncoder()
lr=LinearRegression()
oh=LabelEncoder()
mn=StandardScaler()
lr1=LogisticRegression(multi_class="ovr")
pf=PolynomialFeatures(degree=10)
dataset=pd.read_csv("/Users/mitanshujanghu/Downloads/datasets/coin.csv")
dataset["Date"]=oe.fit_transform(dataset[["Date"]])
dataset["Volume"]=oe.fit_transform(dataset[["Volume"]])
dataset["Market Cap"]=oh.fit_transform(dataset[["Market Cap"]])
give=dataset.iloc[:, :-1]
la=Lasso(alpha=0.001)
ri=Ridge(alpha=100)
for keys in dataset:
    dataset[keys]=mn.fit_transform(dataset[[keys]])
out=dataset["Market Cap"]
pf.fit_transform(give)
x_train,x_test,y_train,y_test=train_test_split(give,out,test_size=0.25)
lr.fit(x_train,y_train)
# sb.pairplot(data=dataset,hue="Market Cap")
# plt.show()
lr1.fit(x_train,y_train)
ri.fit(x_train,y_train)
la.fit(x_train,y_train)
print(lr.score(x_test,y_test)*100)