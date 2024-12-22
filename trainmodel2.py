import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

if __name__ == '__main__':
    df = pd.read_csv('processed_data_2.csv')

    X = df.drop('gpa', axis=1)
    y = df['gpa']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    cross_val_accuracy = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cross_val_accuracy.mean() * 100:.2f}%")

    X = df.drop(['depression_value', 'depression_label', 'feeling_depressed_or_hopeless'], axis=1)
    y = df['depression_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    cross_val_mse = cross_val_score(rf_regressor, X, y, cv=5, scoring='neg_mean_squared_error')
    cross_val_r2 = cross_val_score(rf_regressor, X, y, cv=5, scoring='r2')

    print(f"Cross-validation Mean Squared Error: {-cross_val_mse.mean():.2f}")
    print(f"Cross-validation R-squared: {cross_val_r2.mean():.2f}")
