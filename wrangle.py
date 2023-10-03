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



# Function to download and load the dataset
def load_dataset():
    dataset_url = 'https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset'
    od.download(dataset_url)
    data_dir = 'breast-cancer-dataset'
    breast_cancer = os.path.join(data_dir, 'breast-cancer.csv')
    df = pd.read_csv(breast_cancer)
    return df

# Function to encode categorical data into numerical
def encode_categorical_data(df):
    labelencoder_Y = LabelEncoder()
    df.iloc[:, 1] = labelencoder_Y.fit_transform(df.iloc[:, 1].values)
    return df

# Function to select and keep specific columns
def keep_selected_columns(df):
    selected_columns = ['diagnosis', 'concave points_mean', 'radius_worst', 'perimeter_worst', 'concave points_worst']
    filtered_df = df[selected_columns]
    return filtered_df

