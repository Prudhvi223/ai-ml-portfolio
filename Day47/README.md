
# Day 47 — Project 3: Train and Tune SMS Spam Detection

## Overview

On Day 47, I trained and tuned a machine learning model for SMS Spam Detection.

The project uses Natural Language Processing (NLP) and TF-IDF feature extraction to classify SMS messages as either ham or spam.

## Tasks Completed

- Prepared the dataset for machine learning.
- Separated input messages and target labels.
- Converted labels into numerical values.
- Split the dataset into training and testing sets.
- Applied TF-IDF vectorization.
- Trained a Logistic Regression baseline model.
- Evaluated the model using classification metrics.
- Tuned the model using GridSearchCV.
- Compared baseline and tuned model performance.
- Saved the trained model and TF-IDF vectorizer.
- Tested the saved model on a new SMS message.

## Machine Learning Workflow

1. Load the SMS dataset.
2. Separate input features and target labels.
3. Split the data into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train the Logistic Regression model.
6. Evaluate the baseline model.
7. Tune the hyperparameter `C` using GridSearchCV.
8. Select the best model based on cross-validation F1 Score.
9. Save the model and vectorizer.
10. Test predictions on new messages.

## Text Feature Extraction

TF-IDF was used to convert SMS messages into numerical features.

The vectorizer was fitted only on the training data and then used to transform the test data.

This helps prevent data leakage during model evaluation.

## Models Used

### Baseline Model

Logistic Regression with the default regularization setting.

### Tuned Model

Logistic Regression tuned using GridSearchCV.

The following values of `C` were tested:

- 0.1
- 1
- 10
- 100

Five-fold cross-validation was used during tuning.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

F1 Score was used as the scoring metric during hyperparameter tuning.

## Saved Files

- `models/spam_classifier.joblib` — Trained Logistic Regression model.
- `models/tfidf_vectorizer.joblib` — Fitted TF-IDF vectorizer.
- `experiment_log.csv` — Baseline and tuned model results.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- GridSearchCV
- Joblib
- Jupyter Notebook

## What I Learned

- How to train a text classification model.
- How TF-IDF converts text into numerical features.
- How to evaluate a classification model.
- How precision, recall, and F1 Score are used in spam detection.
- How to tune hyperparameters using GridSearchCV.
- How to save and load a trained machine learning model.
- How to log and compare machine learning experiments.

## Conclusion

Day 47 focused on training and tuning the SMS Spam Detection model.

The tuned model and TF-IDF vectorizer were saved for future use in the FastAPI deployment stage.
