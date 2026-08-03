from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

params = {
    "max_depth": [2, 3, 4, 5],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5
)

grid.fit(X_train, y_train)
print(grid.best_params_)
best_model = grid.best_estimator_

predictions = best_model.predict(X_test)
