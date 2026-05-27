""""
preprocessing.py

Reusable preprocessing pipeline for ML projects.
Handles loading, cleaning, feature engineering,
and train/test splitting.

Author: Anurag Singh
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load dataset from CSV file

    Parameters:
    ----------
    filepath: str
        Path to the CSV file
    
    Returns:
    -------
    pd.DataFrame
        Loaded dataFrame
    """

    # read CSV into dataFrame
    df = pd.read_csv(file_path)

    print(f"Data loaded successfully")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def clean_titanic(df):
    """
    Clean Titanic dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw Titanic dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    #make a copy: since we never modify our original data
    df = df.copy()

    #drop columns with too many missing or redundants
    cols_to_drop = ['deck','embark_town', 'alive', 'class', 'who', 'adult_male']
    df = df.drop(columns=cols_to_drop)

    #fill missing values in 'age' with median
    df['age'] = df['age'].fillna(df['age'].median())

    #fill missing embarked with mode
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    print(f"Titanic data cleaned successfully")
    print(f"Missing values : {df.isnull().sum().sum()}")

    return df


def engineer_titanic(df):
    """
    Feature engineering for Titanic dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned Titanic dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    #make a copy
    df = df.copy()

    #create family size feature
    df['family_size'] = df['sibsp']+df['parch']+1

    #drop original sibsp and parch
    df = df.drop(columns=['sibsp', 'parch', 'alone'])

    #encode sex(binary)
    df['sex'] = df['sex'].map({'male': 0,'female': 1})

    #encode embarked(one-hot)
    df = pd.get_dummies(df, columns=['embarked'], drop_first=True, dtype=int)

    #log transform fare to reduce skewness
    df['fare'] = np.log1p(df['fare'])

    print(f"Titanic features engineered successfully")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


def prepare_for_ml(df,target_col,test_size=0.2,random_state=42):
    """
    Prepare dataframe for ML by splitting into
    train and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature engineered dataframe
    target_col : str
        Name of target column (what we're predicting)
    test_size : float
        Proportion of data for testing (default 0.2 = 20%)
    random_state : int
        Random seed for reproducibility (default 42)
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : pd.DataFrame/Series
        Split features and targets
    """

    #separate features(X) and target(y)
    X=df.drop(columns=[target_col])
    y=df[target_col]

    print(f"Features (X): {X.shape}, Target (y): {y.shape}")

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train/Test split :")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def get_feature_names(df,target_col):
    """
    Get list of feature column names.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    target_col : str
        Target column name to exclude
        
    Returns:
    --------
    list
        Feature column names
    """
    #return all colimns except target
    return [col for col in df.columns if col!=target_col]

def run_titanic_pipeline(filepath):
    """
    Run complete Titanic preprocessing pipeline.
    Convenience function that chains all steps.
    
    Parameters:
    -----------
    filepath : str
        Path to raw Titanic CSV
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """

    print(f"Starting Titanic pipeline ...")
    print("="*40)

    #chain all steps together
    df= load_data(filepath)
    df= clean_titanic(df)
    df= engineer_titanic(df)
    X_train, X_test, y_train, y_test = prepare_for_ml(df,target_col='survived')

    print("="*40)
    print("Pipeline complete. Ready for modeling")

    return X_train, X_test, y_train, y_test