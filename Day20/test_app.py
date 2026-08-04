import joblib


def test_model_load():
    model = joblib.load("churn_model.pkl")
    assert model is not None


def test_model_type():
    model = joblib.load("churn_model.pkl")
    assert model.__class__.__name__ == "LogisticRegression"
