# Spam Email Classifier using Logistic Regression

This project is a basic machine learning pipeline to classify emails as **spam** or **not spam** using **Logistic Regression**. It leverages the Scikit-learn library for model training, evaluation, and visualization.

## 📁 Files

- `spam_classifier.py` - Python script that trains and evaluates a logistic regression model for spam detection.
- `veriler.csv` - CSV file containing the email dataset with features and a `label` column indicating spam (1) or not spam (0).

## 🧠 Objective

To develop a classification model that can distinguish between spam and non-spam (ham) emails based on input features extracted from emails.

## ⚙️ How It Works

1. Loads the dataset from `veriler.csv`.
2. Splits the data into training and testing subsets.
3. Trains a logistic regression model using the training data.
4. Predicts labels for the test set.
5. Prints a classification report (precision, recall, F1-score).
6. Displays a confusion matrix using Matplotlib.

## 🧪 Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib

Install required packages:

```bash
pip install pandas scikit-learn matplotlib
