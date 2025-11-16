import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import sequential_feature_selector
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,LabelEncoder,MinMaxScaler
dataset=pd.read_csv("/Users/mitanshujanghu/Downloads/datasets/data.csv")
edit=OneHotEncoder()
en=edit.fit_transform(dataset[["diagnosis"]])
print(dataset.shape)
print(dataset["area_worst"].describe())
sb.boxplot(x=dataset["area_worst"],data=dataset)
plt.show()
ss=StandardScaler()
ss.fit_transform(dataset[["area_worst"]])
mm=MinMaxScaler()
mm.fit_transform(dataset[["area_se"]])
min=dataset["area_worst"].mean() - 3*dataset["area_worst"].std()
max=dataset["area_worst"].mean() + 3*dataset["area_worst"].std()
# print(min,max)
new_dataset=dataset[dataset["area_worst"]<=max]
dataset.drop_duplicates(inplace=True)
x=dataset.iloc[:,:-1]
y=dataset["area_worst"]
print(new_dataset)