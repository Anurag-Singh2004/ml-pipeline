"""
test_preprocessing.py
---------------------
Tests for src/preprocessing.py functions.
Run with: pytest tests/ -v
"""

import sys
import pytest
import pandas as pd
import numpy as np


sys.path.append('.') #add project to root path

from src.preprocessing import(
    load_data,
    clean_titanic,
    engineer_titanic,
    prepare_for_ml,
    run_titanic_pipeline
)


#FIXTURES : reusable test data
#====================================================

@pytest.fixture
def raw_titanic_df():
    """Load raw titanic data for testing"""
    # using seaborn dataset directly
    import seaborn as sns
    return sns.load_dataset('titanic')

@pytest.fixture
def cleaned_df(raw_titanic_df):
    """Return cleaned titanic dataframe"""
    return clean_titanic(raw_titanic_df)

@pytest.fixture
def engineered_df(cleaned_df):
    """Return feature engineered dataframe"""
    return engineer_titanic(cleaned_df)

#TESTS
#===================================================

def test_load_data(tmp_path):
    """Test load_data function"""
    import seaborn as sns

    #create temporary csv file for testing
    titanic = sns.load_dataset('titanic')
    tmp_file = tmp_path/"test_titanic.csv"
    titanic.to_csv(tmp_file, index=False)

    #test loading
    df = load_data(str(tmp_file))

    #assertions
    assert isinstance(df, pd.DataFrame), "should return DataFrame!"
    assert df.shape[0] == 891, "should have 891 rows!"
    assert df.shape[1] == 15, "should have 15 columns!"
    assert 'survived' in df.columns, "should have survived column!"


def test_clean_titanic(raw_titanic_df):
    """Test clean_titanic function"""

    #ACT
    cleaned = clean_titanic(raw_titanic_df)

    #ASSERT : missing values removed
    assert cleaned.isnull().sum().sum()==0, "no missing values should remain"

    #Assert : redundant columns dropped
    dropped_cols = ['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male']
    for col in dropped_cols:
        assert col not in cleaned.columns, "f{col} should be dropped"
    
    #Asset : correct shape
    assert cleaned.shape[1]==9, "should have 9 columns after cleaning"


def test_engineer_titanic(cleaned_df):
    """Test engineer_titanic function"""

    #ACT
    engineered = engineer_titanic(cleaned_df)

    #Assert : new features created
    assert 'family_size' in engineered.columns, "family_size should be created"

    #Assert : sex encoded correctly
    assert engineered['sex'].isin([0,1]).all(), "sex should be encoded as 0 or 1"

    #Assert : embarked encoded
    assert 'embarked_S' in engineered.columns, "embarked_S should exist after encoding!"

    #Assert: original columns dropped
    assert 'sibsp' not in engineered.columns, "sibsp should be dropped!"

def test_prepare_for_ml(engineered_df):
    """Test prepare_for_ml function"""
    # ACT
    X_train, X_test, y_train, y_test = prepare_for_ml(
        engineered_df, target_col='survived'
    )


    #Assert : correct split sizes
    total = len(engineered_df)
    assert len(X_train) == pytest.approx(total * 0.8, abs=5),  "X_train should be ~80% of data"
    assert len(X_test) == pytest.approx(total * 0.2, abs=5), "X_test should be ~20% of data"

    #Assert : target not found in features
    assert 'survived' not in X_train.columns, "target should not be in features"

    #Assert : shapes match
    assert len(X_train) == len(y_train), "X_train and y_train must have same rows!"

def test_run_Titanic_pipeline(tmp_path):
    """Test complete pipeline end to end"""
    import seaborn as sns

    #save raw data to temp file
    titanic = sns.load_dataset('titanic')
    tmp_file = tmp_path/"test_titanic.csv"
    titanic.to_csv(tmp_file,index=False)

    #Act : run entire pipeline
    X_train, X_test, y_train, y_test = run_titanic_pipeline(
        str(tmp_file)
    )

    #Assert : correct types
    assert isinstance(X_train, pd.DataFrame), "X_train should be DataFrame"
    assert isinstance(X_test, pd.DataFrame),  "X_test should be DataFram"
    assert isinstance(y_train, pd.Series),    "y_train should be Series"
    assert isinstance(y_test, pd.Series),     "y_test should be Series"

    #Assert : correct shapes
    assert X_train.shape[0] == 712, "X_train should have 712 rows"
    assert X_test.shape[0]  == 179, "X_test should have 179 rows"
    assert y_train.shape[0] == 712, "y_train should have 712 rows"
    assert y_test.shape[0]  == 179, "y_test should have 179 rows"

    #Assert : correct number of features
    assert X_train.shape[1] == 7, "should have 7 features"
    assert X_test.shape[1]  == 7, "X_test should have 7 fatures"

    #Assert: no missing values
    assert X_train.isnull().sum().sum() == 0, "X_train has missing values"
    assert X_test.isnull().sum().sum()  == 0, "X_test has missing values"
    assert y_train.isnull().sum()       == 0, "y_train has missing values"
    assert y_test.isnull().sum()        == 0, "y_test has missing values"

    #Assert: target not in feature
    assert 'survived' not in X_train.columns, "target leaked into X_train"
    assert 'survived' not in X_test.columns,  "target leaked into X_test"

    #Assert : target values are binary
    assert y_train.isin([0, 1]).all(), "y_train should only have 0 and 1"
    assert y_test.isin([0, 1]).all(),  "y_test should only have 0 and 1"