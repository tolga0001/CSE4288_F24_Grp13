import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def train_models(X, y, target_columns):
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)
    print(f'KNN - Mean Squared Error: {mse_knn:.4f}')
    print(f'KNN - R-squared: {r2_knn:.4f}')

    dt = DecisionTreeRegressor(random_state=random_state)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    mse_dt = mean_squared_error(y_test, y_pred_dt)
    r2_dt = r2_score(y_test, y_pred_dt)
    print(f'Decision Tree - Mean Squared Error: {mse_dt:.4f}')
    print(f'Decision Tree - R-squared: {r2_dt:.4f}')

    rf = RandomForestRegressor(random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    print(f'Random Forest - Mean Squared Error: {mse_rf:.4f}')
    print(f'Random Forest - R-squared: {r2_rf:.4f}')

def plot_feature_importance(model, feature_names):
    if isinstance(model, DecisionTreeRegressor) or isinstance(model, RandomForestRegressor):
        importance = model.feature_importances_
        feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        feature_importance.plot(kind='bar')
        plt.title('Feature Importance')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()

# Test function to adjust the optimal parameter for the algorithm chosen.
def choose_best_param(X, y, model, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # params must be typed in key-value pair manner as an argument.
    grid_search = GridSearchCV(model, param_grid=params, cv=10, scoring='f1_macro', n_jobs=4)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_ 
    # Displaying the optimal parameter combination
    print(best_params)

# It aims to discard the outliers from the dataset to tune the performance of the algorithm
def pca_analysis(df):
    # Exclude the target column (assuming it's the last column) and standardize the data
    X_scaled = df.iloc[:, :-1]
    # Fit PCA model
    pca = PCA(n_components=2)  # Reducing to 2 components for visualization
    X_pca = pca.fit_transform(X_scaled)
    # Calculate the distance from the center (origin) in PCA space
    distances = np.linalg.norm(X_pca, axis=1)
    # Define a threshold for outliers (e.g., 95th percentile)
    threshold = np.percentile(distances, 80)
    # Identify outliers
    outliers = distances > threshold
    # Plotting PCA result and outliers
    plt.scatter(X_pca[:, 0], X_pca[:, 1], label="Data Points", color="blue")
    plt.scatter(X_pca[outliers, 0], X_pca[outliers, 1], label="Outliers", color="red")
    plt.title("PCA: Outlier Detection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()
    # Print the outlier indices
    print("Outlier Indices:", np.where(outliers)[0])
    # Return the dataset without outliers
    df_no_outliers = df[~outliers]
    return df_no_outliers



if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)

    df = pd.read_csv('processed_data.csv')

    target_column = ['stress_level']
    x = df.drop(columns=target_column)
    y = df[target_column].values.ravel()
    print(df['stress_level'].describe())

    train_models(x, y, target_column)
