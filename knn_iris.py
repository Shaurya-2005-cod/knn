# knn_iris.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
df = pd.read_csv("Iris.csv")
print("First 5 rows of the dataset:\n", df.head())

# Step 2: Drop Unnecessary Column
df.drop("Id", axis=1, inplace=True)

# Step 3: Feature and Target Split
X = df.drop("Species", axis=1)
y = df["Species"]

# Step 4: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train and Evaluate for Multiple K values
accuracies = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K = {k}, Accuracy = {acc:.2f}")

# Step 7: Best K
best_k = accuracies.index(max(accuracies)) + 1
print(f"\nBest K: {best_k}")

# Step 8: Confusion Matrix & Report
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred = best_knn.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), accuracies, marker='o', linestyle='-')
plt.title('K vs Accuracy')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(range(1,11))
plt.show()
