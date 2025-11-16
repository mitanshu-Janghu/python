import pandas as pd
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression,Perceptron
import seaborn as sb
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
import random
import os
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
dataset=pd.read_csv("/Users/mitanshujanghu/shares.csv")
ss=StandardScaler()
oe=OrdinalEncoder()
x=dataset.iloc[:,:-1]
y=dataset["profite ??"]
x.drop('name', axis=1, inplace=True)
x=ss.fit_transform(x)
dt=DecisionTreeClassifier(max_depth=4)
pr=Perceptron()
s=Sequential()
s.add(Dense(6,input_dim=6,activation="relu"))
s.add(Dense(4,activation="relu"))
s.add(Dense(3,activation="relu"))
s.add(Dense(2,activation="relu"))
s.add(Dense(1,activation="sigmoid"))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
dt.fit(x_train,y_train)
s.compile(optimizer="adam",loss="binary_crossentropy")
s.fit(x_train,y_train,batch_size=3,epochs=1000,callbacks=EarlyStopping())
ps=s.predict(x_test)
prd=[]
for i in ps:
    if i[0]>0.5:
        prd.append(1)
    else:
        prd.append(0)
print(accuracy_score(y_test,prd)*100)
row=pd.read_csv("/Users/mitanshujanghu/trail.csv")
jj=[]
je=s.predict(row)
for i in je:
    if i[0]>0.5:
        jj.append(1)
    else:
        jj.append(0)
print(jj)