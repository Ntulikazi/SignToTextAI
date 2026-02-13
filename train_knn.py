import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("hand_detect.csv", header=None)

X = df.iloc[:, :-1].values   
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# single_sample = X_test[0].reshape(1, -1)
# prediction = knn.predict(single_sample)

# print("Predicted gesture:", prediction[0])
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs("models", exist_ok=True)

joblib.dump(knn, "models/knn_models.pkl")

print("Models saved")
