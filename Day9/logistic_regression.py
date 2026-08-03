from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

print("\nActual vs Predicted")

for actual, predicted in zip(y_test, predictions):
    print(
        "Actual:",
        actual,
        "Predicted:",
        predicted
    )

cm = confusion_matrix(y_test, predictions)

print(cm)
