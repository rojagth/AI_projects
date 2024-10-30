import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

print("Script started...")
new_flower = np.array([[5.1, 3.5, 1.4, 5]])

iris = load_iris()
X = iris.data
y = iris.target


print("Features:", iris.feature_names)
print("Target classes:", iris.target_names)
print(pd.DataFrame(X, columns=iris.feature_names).head())  # Show first 5 rows


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

scores = cross_val_score(knn, X, y, cv=5)
print("Cross-validation scores:", scores)
print(f"Average Cross-validation Accuracy: {np.mean(scores) * 100:.2f}%")


y_pred = knn.predict(X_test)
predicted_species = knn.predict(new_flower)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset - Sepal Length vs Sepal Width')
plt.show()

print(f"MY NEW SPECIAL FLOWER IS NUMERO: {predicted_species[0]}")
