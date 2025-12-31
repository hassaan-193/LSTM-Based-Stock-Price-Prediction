import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load dataset
df = pd.read_csv("car_data.csv")

# Drop non-numeric or identifier columns
df = df.drop(['name'], axis=1)

# One-hot encoding (for categorical columns)
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Feature importance (optional, for visualization)
model = ExtraTreesRegressor()
model.fit(X, y)
print("Feature Importances:", model.feature_importances_)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial training
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Evaluation
y_pred = rf_model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot distribution of residuals
sns.histplot(y_test - y_pred, kde=True)
plt.title("Prediction Residuals")
plt.xlabel("Error")
plt.show()

# Hyperparameter tuning with GridSearchCV
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(rf_model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# Save the trained best model
with open('Car_Price_Predicting_System.pkl', 'wb') as file:
    pickle.dump(grid.best_estimator_, file)

# Save feature column names for frontend compatibility
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
