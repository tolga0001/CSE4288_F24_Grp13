import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def train_models(x, y, col_name):
    # Split into test and train data(train->0,7, test->0,3)
    # reproducible random root
    random_root = 12
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_root)

    # Fit and transform for KNN
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    knn_n = 6

    # initialize models

    decision_tree_model = DecisionTreeClassifier(random_state=random_root)
    knn_model = KNeighborsClassifier(knn_n)

    # train models

    decision_tree_model.fit(x_train, y_train)
    knn_model.fit(X_train_scaled, y_train)

    # Predict

    y_prediction_dt = decision_tree_model.predict(x_test)
    y_prediction_knn = knn_model.predict(X_test_scaled)

    # Get scores

    f1_decision_tree = f1_score(y_test, y_prediction_dt, average='weighted')
    f1_knn = f1_score(y_test, y_prediction_knn, average='weighted')

    precision_decision_tree = precision_score(y_test, y_prediction_dt, average='weighted')
    precision_knn = precision_score(y_test, y_prediction_knn, average='weighted')

    recall_decision_tree = recall_score(y_test, y_prediction_dt, average='weighted')
    recall_knn = recall_score(y_test, y_prediction_knn, average='weighted')

    accuracy_decision_tree = accuracy_score(y_test, y_prediction_dt)
    accuracy_knn = accuracy_score(y_test, y_prediction_knn)

    print(f"\n'{col_name}' Scores\n")
    print(f"Decision tree f1 score: {f1_decision_tree:.3f}")
    print(f"KNN f1 score: {f1_knn:.3f}")
    print(f"Decision tree precision: {precision_decision_tree:.3f}")
    print(f"KNN precision: {precision_knn:.3f}")
    print(f"Decision tree recall: {recall_decision_tree:.3f}")
    print(f"KNN recall: {recall_knn:.3f}")

    # Cross validate
    cv_scores_dt = cross_val_score(decision_tree_model, x, y, scoring='accuracy')
    print("Cross-validated accuracy for Decision Tree:", np.mean(cv_scores_dt))

    cv_scores_knn = cross_val_score(knn_model, x, y, scoring='accuracy')
    print("Cross-validated accuracy for KNN:", np.mean(cv_scores_knn))

if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)

    df = pd.read_csv('processed_data.csv')

    target_columns = ['depression', 'anxiety', 'treatment', 'panic_attack']

    for col_name in target_columns:
        y = df[col_name]
        x = df.drop(columns=target_columns)  #df without targets

        train_models(x, y, col_name)






