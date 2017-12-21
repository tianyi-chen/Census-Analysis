# General libraries
import pandas as pd
import numpy as np

# Libraries for model training
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

def clean(df):
    '''
    Cleans the data
    '''
    # Remove empty values
    df = df[~df.isnull().any(axis=1)]
    
    df_y = df['IncomeBracket']
    df_X = df.drop(labels=['IncomeBracket'], axis=1)
    
    # Convert the target into numerical
    dic1 = {"<50K": 0, "50-100K": 1, ">100K": 2}
    df_y = df_y.map(dic1)
    
    # Remove Education and HoursPerWeek
    df_X = df_X.drop(labels=['Education', 'HoursPerWeek'], axis=1)
    # Create NetGain and remove CapitalGian and CapitalLoss
    net_gain = df_X['CapitalGain'] - df_X['CapitalLoss']
    df_X.insert(loc=3, column='NetGain', value=net_gain)
    df_X = df_X.drop(labels=['CapitalGain', 'CapitalLoss'], axis=1)
    # Convert MaritalStatus
    dic2 = {"Married-civ-spouse": 1, "Divorced": 0, "Never-married": 0, "Separated": 1,
       "Widowed": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 1}
    df_X['MaritalStatus'] = df_X['MaritalStatus'].map(dic2)
    # 
    dic3 = {"United-States": 1}
    df_X['NativeCountry'] = df_X['NativeCountry'].map(dic3)
    df_X['NativeCountry'].fillna(0, inplace=True)
    
    #Convert the feature variables into numerical
    df_X = pd.get_dummies(df_X, drop_first=True)
    
    return df_X, df_y
    
def fix_columns(columns, df): 
    '''
    Fix the columns to make sure df has the same number of columns as the model requires
    '''
    # Add missing dummy columns
    missing_cols = set(columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0

    # Make sure we have all the columns we need
    assert(set(columns) - set(df.columns) == set())

    extra_cols = set(df.columns) - set(columns)
    if extra_cols:
        print("extra columns:", extra_cols)

    df = df[columns]
    return df
    
def bonus(training_file_path, testing_file_path):
    # Clean the training data
    training = pd.read_csv(training_file_path)
    X_train, y_train = clean(training)
        
    # Train the model with optimal parameter
    gb = GradientBoostingClassifier(n_estimators=100)
    gb.fit(X_train, y_train)
        
    # Clean the testing data
    testing = pd.read_csv(testing_file_path)
    X_test, y_test = clean(testing)
    X_test = fix_columns(X_train.columns, X_test)
        
    # Predict the labels of testing data
    predictions = gb.predict(X_test)
    dic = {0: "<50K", 1: "50-100K", 2: ">100K"}
    income_brackets = pd.DataFrame({'IncomeBracket':predictions.tolist()})['IncomeBracket'].map(dic)
        
    # Write predicted labels in csv file
    #np.savetxt("chen_1003908630_assignment2_bonus.csv", predictions, fmt='%i', delimiter=',', header='IncomeBracket')
    np.savetxt("chen_1003908630_assignment2_bonus.csv", income_brackets, fmt='%s', delimiter=',', header='IncomeBracket')
