from enum import Enum 

class PathEnum(str, Enum): 
    TRAIN_PATH = '/mnt/e/Machine Learning/NYC_Taxi_Trip_Duration/src/data/split/train.csv' 
    VAL_PATH = '/mnt/e/Machine Learning/NYC_Taxi_Trip_Duration/src/data/split/val.csv' 
    TEST_PATH = '/mnt/e/Machine Learning/NYC_Taxi_Trip_Duration/src/data/split/test.csv'
    API_TEST_PATH = '/mnt/e/Machine Learning/NYC_Taxi_Trip_Duration/src/data/api/test.csv'
    