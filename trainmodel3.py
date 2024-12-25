import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Random State for Reproducibility
random_state = 12

# Load dataset
df = pd.read_csv('Updated_Student_Mental_Health_Weighted.csv')

# Check for missing values (Basic Data Preprocessing)
if df.isnull().sum().any():
    print("Warning: Dataset contains missing values.")
    df = df.dropna()  # Handle missing values by dropping rows (optional)

# KNN Classifier
print("\nKNN for GPA Prediction\n")

# Define features (X) and target (y) for GPA prediction
X = df.drop('gpa', axis=1)
y = df['gpa']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# Best hyperparameters from manual tuning or GridSearchCV
best_params_knn = {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'manhattan', 'n_neighbors': 1, 'weights': 'uniform'}

# Initialize KNN model with best hyperparameters
knn_model = KNeighborsClassifier(**best_params_knn)
knn_model.fit(X_train, y_train)

# Evaluate KNN model
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
cross_val_knn = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')

print(f"Accuracy: {accuracy_knn * 100:.2f}%")
print(f"Cross-validation Accuracy: {cross_val_knn.mean() * 100:.2f}%")

# Random Forest Regressor
print("\nRandom Forest for Health Score Value Prediction\n")

# Define features (X) and target (y) for Depression Value prediction
X = df.drop([ 'depression_label','depression_value','Health_Score'], axis=1)
y = df['Health_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# Best hyperparameters for Random Forest
best_params_rf = {
    'bootstrap': False, 'criterion': 'squared_error', 'max_depth': 30, 'max_features': 'sqrt',
    'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1
}

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(**best_params_rf)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest model
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
cross_val_mse_rf = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_r2_rf = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

print(f"Mean Squared Error: {mse_rf:.2f}")
print(f"R-squared: {r2_rf:.2f}")
print(f"Cross-validation MSE: {-cross_val_mse_rf.mean():.2f}")
print(f"Cross-validation R-squared: {cross_val_r2_rf.mean():.2f}")

# Gradient Boosting Regressor
print("\nGradient Boosting Regressor for Health Score Value Prediction\n")

# Initialize Gradient Boosting Regressor
gboost_model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
gboost_model.fit(X_train, y_train)

# Evaluate Gradient Boosting Regressor
y_pred_gboost = gboost_model.predict(X_test)
mse_gboost = mean_squared_error(y_test, y_pred_gboost)
r2_gboost = r2_score(y_test, y_pred_gboost)
cross_val_mse_gboost = cross_val_score(gboost_model, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_r2_gboost = cross_val_score(gboost_model, X, y, cv=5, scoring='r2')

print(f"Mean Squared Error: {mse_gboost:.2f}")
print(f"R-squared: {r2_gboost:.2f}")
print(f"Cross-validation MSE: {-cross_val_mse_gboost.mean():.2f}")
print(f"Cross-validation R-squared: {cross_val_r2_gboost.mean():.2f}")

# Now we plot the comparison of R-squared and MSE for each model
algorithms = ['Random Forest', 'Gradient Boosting']
r2_scores = [r2_rf, r2_gboost]
mse_values = [mse_rf, mse_gboost]

# Create subplots for comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot for R-squared scores
ax[0].bar(algorithms, r2_scores, color=['green', 'orange'])
ax[0].set_title('R-squared Comparison')
ax[0].set_ylabel('R-squared')

# Plot for Mean Squared Errors
ax[1].bar(algorithms, mse_values, color=['green', 'orange'])
ax[1].set_title('Mean Squared Error Comparison')
ax[1].set_ylabel('MSE')

plt.tight_layout()
plt.show()