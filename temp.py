import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
row=pd.read_csv("/Users/mitanshujanghu/trail.csv")
ss=StandardScaler()
mn=MinMaxScaler()
row=ss.fit_transform(row)
print(row)