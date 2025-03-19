#!/home/aws/miniconda3/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.linalg as lng
from scipy.spatial import distance
from sklearn import preprocessing as preproc
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from tqdm import tqdm  # Import tqdm for progress bars

# Data loading
print("Loading data...")
data = np.genfromtxt('../data/case1Data.csv', delimiter=',', skip_header=1, filling_values=np.nan)
y = data[:, 0]
X = data[:, 1:]

# Wrangling
df = pd.read_csv("../data/case1Data.csv")
df_new = pd.read_csv("../data/case1Data_Xnew.csv")

print("Performing data wrangling...")
# Mode imputation for the categorical features
df.iloc[:, -5:] = df.iloc[:, -5:].apply(lambda col: col.fillna(col.mode()[0]))
df_new.iloc[:, -5:] = df_new.iloc[:, -5:].apply(lambda col: col.fillna(col.mode()[0]))

# Filling missing values in numerical features with the mean
df.iloc[:, 1:-5] = df.iloc[:, 1:-5].apply(lambda col: col.fillna(col.mean()))

# One-hot encoding the categorical features
df = pd.get_dummies(df, columns=df.columns[-5:], dtype=int)

# Convert numerical columns to float before applying standardization
df.iloc[:, :-36] = df.iloc[:, :-36].astype(float)

# Prepare the features and target
y = df.iloc[:, 0].to_numpy()
X = df.iloc[:, 1:].to_numpy()

# Train-test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the numerical features
print("Standardizing the data...")
scaler = preproc.StandardScaler()
X_train = scaler.fit_transform(X_train[:, :-21])
X_test = scaler.transform(X_test[:, :-21])
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

## Polynomial Features (if non-linearity is suspected)
#print("Generating polynomial features...")
#poly = PolynomialFeatures(degree=2)  # You can adjust the degree based on experimentation
#X_train_poly = poly.fit_transform(X_train)
#X_test_poly = poly.transform(X_test)

# Feature Selection (Recursive Feature Elimination with Cross-Validation)
print("Performing feature selection...")
model_rfe = ElasticNetCV(cv=5, random_state=42)
rfe = RFE(model_rfe, n_features_to_select=20)  # Select top 20 features (adjust as needed)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Initialize ElasticNetCV with cross-validation
print("Training ElasticNetCV model...")
model = ElasticNetCV(cv=5, random_state=42)

# Fit the model to the training data with progress bar
for _ in tqdm(range(50), desc="Fitting ElasticNetCV model"):
    model.fit(X_train_rfe, y_train)

# Print the best alpha and l1_ratio found
print("Optimal alpha:", model.alpha_)
print("Optimal l1_ratio:", model.l1_ratio_)

# Evaluate the model on the test set
print("Evaluating model on the test set...")
test_score = model.score(X_test_rfe, y_test)
print("Test set R^2 score:", test_score)

# Predict the target values
print("Making predictions...")
y_pred = model.predict(X_test_rfe)

# Calculate RMSE
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print("RMSE:", rmse)

# Optionally, try a different model such as Random Forest to compare results
print("Training Random Forest model...")
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the Random Forest model
for _ in tqdm(range(1), desc="Fitting Random Forest model"):
    rf_model.fit(X_train_rfe, y_train)

# Predict with Random Forest
rf_y_pred = rf_model.predict(X_test_rfe)

# Calculate RMSE for Random Forest
rf_rmse = np.sqrt(np.mean((y_test - rf_y_pred)**2))
print("Random Forest RMSE:", rf_rmse)

