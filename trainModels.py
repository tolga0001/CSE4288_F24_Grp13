import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def train_models(X, y, target_columns):
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

def plot_confusion_matrix(y_true, y_pred, model_name):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title(f'Confusion Matrix for {model_name} - Predicted vs Actual')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

def plot_feature_importance(model, feature_names):
    # Only works for tree-based models (like DecisionTreeClassifier)
    if isinstance(model, DecisionTreeClassifier):
        importance = model.feature_importances_
        feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        feature_importance.plot(kind='bar')
        plt.title('Feature Importance - Decision Tree Model')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.show()

if __name__ == '__main__':
    pd.set_option('display.max_columns', 20)

    df = pd.read_csv('processed_data.csv')

    target_column = ['depression']
    x = df.drop(columns=target_column)
    y = df[target_column].values.ravel()
    print(df['depression'].value_counts())

    train_models(x, y, target_column)
