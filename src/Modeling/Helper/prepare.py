'''

'''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import pandas as pd 
from data.load_data import load_data, load_X_t, split_data
from Enum.path_enums import PathEnum as penum
from Enum.feats_enums import FeatureEnum as fenum  
from Preprocessing.preprocessing import Preprocessing_Pipline
train_data_path=penum.TRAIN_PATH.value
val_data_path=penum.TEST_PATH.value 

class Preparing(): 
    def __init__(self, train_path=train_data_path): 
        self.train = load_data(train_path) 
        self.val = load_data(val_data_path) 
        self.preprocessing_pipline = Preprocessing_Pipline() 

    def prepare_data(self): 
        self.train, lable_encoder_season, lable_encoder_store, hdt, outliayerlimts = self.preprocessing_pipline.fit_transform(self.train) 
        self.val = self.preprocessing_pipline.transform(self.val, lable_encoder_season=lable_encoder_season, lable_encoder_store=lable_encoder_store, hdt=hdt, outliayerLimits=outliayerlimts) 

        t_train = self.train[fenum.LOG_TRIP_DURATION.value] 
        t_val = self.val[fenum.LOG_TRIP_DURATION.value] 

        x_train = self.train.drop(columns=[fenum.LOG_TRIP_DURATION.value]) 
        x_val = self.val.drop(columns=[fenum.LOG_TRIP_DURATION.value]) 

        poly, x_train, x_val = self.preprocessing_pipline.polynomial_feature(x=x_train, x_val=x_val)

        scaler, x_train, x_val = self.preprocessing_pipline.scaling(x=x_train, x_val=x_val) 
        return x_train, x_val, t_train, t_val, lable_encoder_season, lable_encoder_store, poly, scaler, hdt, outliayerlimts 




if __name__ == '__main__': 
    prepare = Preparing() 
    x, x_v, t, t_v, es, est, p, s=prepare.prepare_data()
    print(x.shape)
    print(x_v.shape)
    print(t.shape)
    print(t_v.shape)
