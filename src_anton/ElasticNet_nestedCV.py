import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

# Example dataset (replace this with your own dataset)


# Path to the data files
data_path_1 = '../data/case1Data.csv'

# Load the data into a numpy array
data_np = np.loadtxt(data_path_1, delimiter=',', skiprows=1)

# Create a pandas dataframe and use the first row as the column names
data_pd = pd.read_csv(data_path_1, sep=',', header=0)


# Splitting the data into features and target
X = data_pd.iloc[:, 1:].to_numpy()
y = data_pd.iloc[:, 0].to_numpy()

print("X: ", X.shape)
print("y: ", y.shape)


# Define which columns are continuous and categorical
# Example: assuming all features are continuous
# If you have categorical columns, update accordingly
categorical_cols = list(range(X.shape[1]-5, X.shape[1]))  # Last 5 columns as categorical
continuous_cols = list(range(X.shape[1] - 5))  # All columns before the last 5 as continuous

# Define preprocessing steps for continuous and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),  # KNN Imputer for continuous features
            ('scaler', StandardScaler())  # Standardization of continuous features
        ]), continuous_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute categorical missing values
            ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One hot encode categorical features
        ]), categorical_cols)
    ]
)

# Define the classifier (example with RandomForest)
regressor = ElasticNet()

# Create a pipeline that first applies preprocessing and then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

# Nested Cross-Validation (Outer loop for model evaluation, Inner loop for hyperparameter tuning)
# Inner loop for hyperparameter tuning
param_grid = {
    'alpha': [0.1, 1, 10, 100],  # Regularization strength
    'l1_ratio': [0.1, 0.5, 0.9]  # Mix between Lasso and Ridge
}

# Outer loop for cross-validation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV for hyperparameter tuning (inner loop)
grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=-1)

# Evaluate with cross_val_score (outer loop)
nested_score = cross_val_score(grid_search, X, y, cv=outer_cv, n_jobs=-1)

# Print results
print(f"Nested CV score: {nested_score.mean()} ± {nested_score.std()}")
