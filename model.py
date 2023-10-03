import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import opendatasets as od
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb



# Function to train and evaluate models
def train_and_evaluate_models(X_train, Y_train, X_validation, Y_validation):
    models = []
    
    # Logistic Regression
    log = LogisticRegression(random_state=42)
    log.fit(X_train, Y_train)
    models.append(log)

    # Decision Tree Classifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    tree.fit(X_train, Y_train)
    models.append(tree)

    # Random Forest Classifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=42)
    forest.fit(X_train, Y_train)
    models.append(forest)
    
    # K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train, Y_train)
    models.append(knn)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42)
    xgb_model.fit(X_train, Y_train)
    models.append(xgb_model)
    
    # Evaluate models on train set
    for idx, model in enumerate(models):
        train_predictions = model.predict(X_train)
        train_accuracy = accuracy_score(Y_train, train_predictions)
        print(f'Model [{idx}] Train Accuracy: {train_accuracy:.2f}')
        
        # Evaluate models on validation set
    for idx, model in enumerate(models):
        validation_predictions = model.predict(X_validation)
        validation_accuracy = accuracy_score(Y_validation, validation_predictions)
        print(f'Model [{idx}] Validation Accuracy: {validation_accuracy:.2f}')

    return models






# Function to select and evaluate the best model
def evaluate_best_model(models, X_test, Y_test):
    selected_model = models[0]
    test_predictions = selected_model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    predictions_df = pd.DataFrame({'Predictions': test_predictions, 'True Labels': Y_test})
    csv_file_path = 'predictions.csv'
    predictions_df.to_csv(csv_file_path, index=False)
    print(f'Predictions saved to {csv_file_path}')