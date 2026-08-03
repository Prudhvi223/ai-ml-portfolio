import category_encoders as ce

encoder = ce.TargetEncoder(cols=["City"])

X_train["City"] = encoder.fit_transform(
    X_train["City"],
    y_train
)

X_test["City"] = encoder.transform(
    X_test["City"]
)
