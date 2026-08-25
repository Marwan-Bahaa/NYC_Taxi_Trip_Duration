from sklearn.model_selection import train_test_split 
import pandas as pd 
from Config.load import load_config 

config = load_config()


def load_data(path:str): 
    if path is None: 
        return ValueError('This is invalid path')  
    df = pd.read_csv(path) 
    return df 



def load_X_t(df:pd.DataFrame): 
    t = df.iloc[:,-1] 
    X = df.iloc[:,:-1]  
    return X, t 
 

def split_data(X, t, s_size=0.2): 
    train_test_split(X, t, test_size=s_size, random_state=config['RANDOM_STATE'])

