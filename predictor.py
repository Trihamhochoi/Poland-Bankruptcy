# Import libraries
import gzip
import json
import pickle
from scipy.io import arff
import pandas as pd
import re
import numpy as np


# Add wrangle function from lesson 5.4
def wrangle(filename):
    
    # Open compressed file, load into dictionary
    data = arff.loadarff(filename)[0]
        
    # Load dictionary into DataFrame, set index
    df = pd.DataFrame(data=data)
    
    # Rename columns
    df.columns = [re.sub(pattern='Attr',repl='feat_',string=col) for col in df.columns]
    df.rename(columns={'class':'bankrupt'},inplace=True)
    
    # Change dtype of the Labels columns
    df['bankrupt'] = df['bankrupt'].astype(np.int64)
    
    # column is the most missing value
    df.drop(columns='feat_37',inplace=True)
    return df


def make_predictions(data_filepath, model_filepath):
    # Wrangle JSON file
    df = wrangle(data_filepath)
    
    # Split to X and y 
    target = "bankrupt"
    X_test = df.drop(columns=target)
    
    # Load model
    with open(model_filepath,'rb') as f:
        model = pickle.load(f)
        
    # Generate predictions
    y_test_pred = model.predict(X_test)
    
    # Put predictions into Series with name "bankrupt", and same index as X_test
    y_test_pred = pd.Series(y_test_pred,index=X_test.index,name='bankrupt')
    
    return y_test_pred