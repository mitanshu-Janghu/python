import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7],
    'Previous_Score': [40, 50, 60, 70, 80, 85, 90],
    'Pass': [0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Previous_Score']]
y = df['Pass']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier(
    criterion="gini",   
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))