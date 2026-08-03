from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

X = iris.data
y = iris.target

model = DecisionTreeClassifier(random_state=42)

params = {
    "max_depth": [2, 3, 4, 5, 6, 7, 8],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 4, 6, 8]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=10,      # Try only 10 random combinations
    cv=5,
    random_state=42
)

random_search.fit(X, y)
