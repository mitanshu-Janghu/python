import pandas as pd
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Perceptron
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
dataset=pd.read_csv("/Users/mitanshujanghu/shares.csv")
ss=StandardScaler()
x=dataset.iloc[:,:-1]
y=dataset["profite ??"]
x.drop('name', axis=1, inplace=True)
x=ss.fit_transform(x)
s=Sequential()
# s.add(Dense(6,input_dim=6,activation="relu"))
# s.add(Dense(4,activation="relu"))
# s.add(Dense(3,activation="relu"))
# s.add(Dense(2,activation="relu"))
s.add(Dense(1,activation="sigmoid"))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
s.compile(optimizer="adam",loss="binary_crossentropy")
s.fit(x_train,y_train,batch_size=17,epochs=1,callbacks=EarlyStopping())
ps=s.predict(x_test)
prd=[]
for i in ps:
    if i[0]>0.5:
        prd.append(1)
    else:
        prd.append(0)
row=pd.read_csv("/Users/mitanshujanghu/trail.csv")
jj=[]
je=s.predict(row)
for i in je:
    if i[0]>0.5:
        jj.append(1)
    else:
        jj.append(0)
print(jj)