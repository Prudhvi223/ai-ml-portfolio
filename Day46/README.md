
# Day 46 — Project 3 Kickoff: SMS Spam Detection

## Overview

On Day 46, I started Project 3 of my AI/ML roadmap.

The project focuses on building an SMS Spam Detection system using Natural Language Processing (NLP) and Machine Learning.

The goal is to classify SMS messages into two categories:

- Ham: Legitimate messages
- Spam: Unwanted or suspicious messages

## Dataset

I used the SMS Spam Collection dataset.

The dataset contains SMS messages and their corresponding labels.

### Dataset Columns

- `label`: Target category (ham or spam)
- `message`: SMS text

## Topics Covered

- Dataset downloading and loading
- Exploratory Data Analysis (EDA)
- Label distribution
- Missing value checking
- Duplicate value checking
- Message length analysis
- Word count analysis
- Dataset preparation

## Exploratory Data Analysis

During EDA, I performed the following tasks:

1. Loaded the dataset using Pandas.
2. Checked the dataset shape and structure.
3. Checked for missing values.
4. Analyzed the distribution of ham and spam messages.
5. Removed duplicate rows.
6. Calculated message length.
7. Calculated word count.
8. Visualized message length and word count distributions.
9. Saved the EDA dataset as a CSV file.

## Features Created

### Message Length

The number of characters in each SMS message.

### Word Count

The number of words in each SMS message.

These features were created for exploratory analysis.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Scikit-learn

## Project Workflow

The planned workflow for the project is:

1. Dataset collection
2. Exploratory Data Analysis
3. Text preprocessing
4. TF-IDF feature extraction
5. Model training
6. Hyperparameter tuning
7. Model evaluation
8. FastAPI deployment
9. GitHub documentation

## What I Learned

- How to load a text classification dataset.
- How to analyze label distribution.
- How to check missing and duplicate values.
- How to calculate message length and word count.
- How to visualize text-related features.
- How to prepare a dataset for machine learning.

## Conclusion

Day 46 marked the beginning of Project 3.

I explored the SMS Spam Collection dataset and prepared the data for model training and tuning in Day 47.
