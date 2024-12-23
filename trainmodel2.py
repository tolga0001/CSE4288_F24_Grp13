import pandas as pd
from matplotlib import pyplot as plt
from scipy.constants import grain
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    #Best Hyperparameters For KNN :=
    # {'algorithm': 'auto', 'leaf_size': 10, 'metric': 'manhattan', 'n_neighbors': 1, 'p': 1, 'weights': 'uniform'}

    grid_knn = param_grid_knn = {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
        'p': [1, 2, 3],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50, 60],
    }

    best_params = {
        'algorithm': 'auto', 'leaf_size': 10, 'metric': 'manhattan', 'n_neighbors': 1, 'p': 1, 'weights': 'uniform'
    }

    random_state = 12

    df = pd.read_csv('processed_data_2.csv')

    X = df.drop('gpa', axis=1)
    y = df['gpa']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    #knn_classifier = KNeighborsClassifier()

    #grid_search_knn = GridSearchCV(estimator=knn_classifier, param_grid=param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
    #grid_search_knn.fit(X_train, y_train)

    #best_params = grid_search_knn.best_params_
    #best_model = grid_search_knn.best_estimator_

    best_model = KNeighborsClassifier(
        algorithm=best_params['algorithm'],
        leaf_size=best_params['leaf_size'],
        metric=best_params['metric'],
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
    )

    best_model.fit(X_train, y_train)
    print("Best Hyperparameters For KNN:=", best_params)

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nKNN for gpa\n")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    cross_val_accuracy = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cross_val_accuracy.mean() * 100:.2f}%")

    #
    #
    #
    #
    #random forest
    #
    #
    #
    #
    #Best Hyperparameters For Random Forest:
    # {'bootstrap': False, 'criterion': 'squared_error', 'max_depth': 30, 'max_features': 'sqrt',
    # 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100, 'n_jobs': -1}
    print("\nRandom Forest\n")

    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
        'n_jobs': [-1]
    }

    #from grid search
    best_params = {'bootstrap': False,
                   'criterion': 'squared_error',
                   'max_depth': 30,
                   'max_features': 'sqrt',
                   'min_samples_leaf': 1,
                   'min_samples_split': 2,
                   'n_estimators': 100,
                   'n_jobs': -1}

    rf_regressor = RandomForestRegressor(
        bootstrap=best_params['bootstrap'],
        criterion=best_params['criterion'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        n_estimators=best_params['n_estimators'],
        n_jobs=best_params['n_jobs']
    )

    X = df.drop(['depression_value', 'depression_label', 'feeling_depressed_or_hopeless'], axis=1)
    y = df['depression_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    rf_regressor.fit(X_train, y_train)
    #rf_regressor = RandomForestRegressor()

    #grid_search_rf = GridSearchCV(estimator=rf_regressor, param_grid=param_grid_rf, cv=5,
                                  #scoring='neg_mean_squared_error', n_jobs=-1)
    #grid_search_rf.fit(X_train, y_train)



    #best_params = grid_search_rf.best_params_
    #best_model_rf = grid_search_rf.best_estimator_

    print("Best Hyperparameters For Random Forest:", best_params)

    #y_pred = best_model_rf.predict(X_test)
    y_pred = rf_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    cross_val_mse = cross_val_score(rf_regressor, X, y, cv=5, scoring='neg_mean_squared_error')
    cross_val_r2 = cross_val_score(rf_regressor, X, y, cv=5, scoring='r2')

    print(f"Cross-validation Mean Squared Error: {-cross_val_mse.mean():.2f}")
    print(f"Cross-validation R-squared: {cross_val_r2.mean():.2f}")
    #
    #
    #
    #
    #
    #Gradient boost
    #
    #
    #
    #
    #
    print("\nGradient Boosting Regressor\n")

    X = df.drop(['depression_value', 'depression_label', 'feeling_depressed_or_hopeless'], axis=1)
    y = df['depression_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    gradient_boost = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    gradient_boost.fit(X_train, y_train)

    y_pred = gradient_boost.predict(X_test)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    cross_val_mse_g_boost = cross_val_score(gradient_boost, X, y, cv=5, scoring='neg_mean_squared_error')
    cross_val_r2_g_boost = cross_val_score(gradient_boost, X, y, cv=5, scoring='r2')

    print(f"Cross-validation Mean Squared Error: {-cross_val_mse_g_boost.mean():.2f}%")
    print(f"Cross-validation R-squared: {cross_val_r2_g_boost.mean():.2f}%")





