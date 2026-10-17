import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    classification_report,

    confusion_matrix

)

model = MultinomialNB()


df = pd.read_csv("emails.csv")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
x=df["text"]
y=df["spam"]
x = vectorizer.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(

    x, y, test_size=0.2, random_state=42

)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred, zero_division=0))

print("Recall:", recall_score(y_test, y_pred, zero_division=0))

print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nClassification Report:")

print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

