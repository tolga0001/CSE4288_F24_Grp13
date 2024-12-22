import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


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

if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)

    df = pd.read_csv('processed_data.csv')

    target_column = ['stress_level']
    x = df.drop(columns=target_column)
    y = df[target_column].values.ravel()
    print(df['stress_level'].describe())

    train_models(x, y, target_column)
