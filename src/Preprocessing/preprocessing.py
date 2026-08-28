import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd 
from data.load_data import load_data, load_X_t, split_data
from Enum.path_enums import PathEnum as penum
from Enum.feats_enums import FeatureEnum as fenum 
from Enum.nyc_land_mark_enums import NYC_Landmark_Enum as lmenum
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, Normalizer, LabelEncoder
from Config.load import load_config
config = load_config()

#enum var
train_path  = penum.TRAIN_PATH.value 
# all_landmarks = lmenum.get_all_landmarks()

#config val
poly_degree = config['preprocessing']['polynomial']['degree']

poly_include_bias = config['preprocessing']['polynomial']['include_bias']

scaling_option = config['preprocessing']['scaling']['option']

drop_outliayer = config['preprocessing']['drop_outliayer'] 

drop_feats = config['preprocessing']['drop_feats']

add_landmarkb = config['preprocessing']['add_landMark'] 

beast_feats = config['preprocessing']['best_feats']

high_dist_thresholdq = config['preprocessing']['haversine_distance']['qthreshold']

land_jfk_lat = config['preprocessing']['nyc_landmarks']['jfk']['latitude'] 
land_jfk_long = config['preprocessing']['nyc_landmarks']['jfk']['longitude'] 

land_lga_lat = config['preprocessing']['nyc_landmarks']['lga']['latitude'] 
land_lga_long = config['preprocessing']['nyc_landmarks']['lga']['longitude'] 

land_ewr_lat = config['preprocessing']['nyc_landmarks']['ewr']['latitude'] 
land_ewr_long = config['preprocessing']['nyc_landmarks']['ewr']['longitude'] 

land_times_square_lat = config['preprocessing']['nyc_landmarks']['times_square']['latitude'] 
land_times_square_long = config['preprocessing']['nyc_landmarks']['times_square']['longitude'] 




class Preprocessing_Pipline(): 
    def __init__(self):
        self.outliayer_limts = {}  # col : (lower,upper)
        self.poly = None
        self.scaler = None 
        self.lable_encoder_season = None 
        self.lable_encoder_store = None    
        self.high_dist_threshold = None 

    def __comput_outliayer_limts(self, df:pd.DataFrame): 
        limits = {}
        columns = df.select_dtypes(np.number).columns
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3-q1
            lower = q1 - 1.5*iqr
            upper = q3 + 1.5*iqr
            limits[col] = (lower, upper)
        return limits
         

    def __apply_outliayer_limts(self, df:pd.DataFrame): 
       for col, (lower, upper) in self.outliayer_limts.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)

       return df         


    def __calc_haversine_distance(self, lat1, lon1, lat2, lon2):
            R = 6371.0
            lat1,lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2-lat1
            dlon = lon2-lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2*R*np.arcsin(np.sqrt(a)) 

    def __add_landmark_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Haversine distances to major NYC landmarks from pickup and dropoff coordinates."""
        df_out = df.copy()
        
        LandMarks = {
            lmenum.JFK.value: (land_jfk_lat, land_jfk_long), 
            lmenum.EWR.value: (land_ewr_lat, land_ewr_long), 
            lmenum.LGA.value: (land_lga_lat, land_lga_long), 
            lmenum.TIMES_SQUARE.value: (land_times_square_lat, land_times_square_long)
        }
        
        # 1. Explicitly list actual airport landmark keys
        airport_names = [lmenum.JFK.value, lmenum.EWR.value, lmenum.LGA.value]
        
        # Calculate distances to landmarks
        for landmark_name, (l_lat, l_lon) in LandMarks.items():
            pickup_col = f'pickup_dist_{landmark_name}'
            dropoff_col = f'dropoff_dist_{landmark_name}'
            
            df_out[pickup_col] = self.__calc_haversine_distance(
                df_out[fenum.PICKUP_LATITUDE.value], df_out[fenum.PICKUP_LONGITUDE.value],
                l_lat, l_lon
            )
            
            df_out[dropoff_col] = self.__calc_haversine_distance(
                df_out[fenum.DROPOFF_LATITUDE.value], df_out[fenum.DROPOFF_LONGITUDE.value],
                l_lat, l_lon
            )
            
            # 2. Only create airport flags for actual airports
            if landmark_name in airport_names:
                df_out[f'is_airport_{landmark_name}'] = (
                    (df_out[pickup_col] < 2.0) | (df_out[dropoff_col] < 2.0)
                ).astype(int)

        # 3. Aggregate using only the generated airport columns
        airport_cols = [f'is_airport_{name}' for name in airport_names]
        df_out[fenum.IS_ANY_AIRPORT.value] = df_out[airport_cols].max(axis=1)
        
        return df_out
            
       
    def fit_transform(self, df:pd.DataFrame): 
        df = df.copy() 

        df[fenum.PICKUP_DATETIME.value] = pd.to_datetime(df[fenum.PICKUP_DATETIME.value]) 
        df[fenum.HOUR.value] = df[fenum.PICKUP_DATETIME.value].dt.hour 
        df[fenum.MONTH.value] = df[fenum.PICKUP_DATETIME.value].dt.month 
        df[fenum.DAY_OF_WEEK.value] = df[fenum.PICKUP_DATETIME.value].dt.dayofweek 

        
        def getseason(month):
            if  4 <= month <= 7: return 'Sprint'
            elif  8 <= month <= 10: return 'Summar'
            elif  11 <= month <= 12: return 'Fail'
            else: return 'Winter ' 


        df[fenum.SEASON.value] = df[fenum.MONTH.value].apply(getseason)   

        self.lable_encoder_season = LabelEncoder()
        df[fenum.SEASON.value] = self.lable_encoder_season.fit_transform(df[fenum.SEASON.value]) 

        self.lable_encoder_store = LabelEncoder() 
        df[fenum.STORE_AND_FWD_FLAG.value] = self.lable_encoder_store.fit_transform(df[fenum.STORE_AND_FWD_FLAG.value])


        df[fenum.IS_WEEKEND.value] = df[fenum.DAY_OF_WEEK.value].apply(lambda x: 1 if x >= 5 else 0)

        # 1. Flag late night hours
        df[fenum.IS_LATE_NIGHT.value] = df[fenum.HOUR.value].isin([20, 21, 22, 23, 0, 1, 2, 3, 4]).astype(int)


        # Binary flag for group vs. solo rides
        df[fenum.IS_GROUP_TRIP.value] = (df[fenum.PASSENGER_COUNT.value] > 1).astype(int) 

        df[fenum.IS_RUSH_HOUR.value] = df[fenum.HOUR.value].isin([7, 8, 9, 16, 17, 18]).astype(int)

        df[fenum.LOG_DISTANCE.value] = np.log1p(self.__calc_haversine_distance(
            lat1=df[fenum.PICKUP_LATITUDE.value], 
            lon1=df[fenum.PICKUP_LONGITUDE.value], 
            lat2=df[fenum.DROPOFF_LATITUDE.value], 
            lon2=df[fenum.DROPOFF_LONGITUDE.value] 
        )) 

        # Highlight long-haul trips (top 15% distance threshold)
        self.high_dist_threshold = df[fenum.LOG_DISTANCE.value].quantile(high_dist_thresholdq)
        df[fenum.IS_LONG_HAUL.value] = (df[fenum.LOG_DISTANCE.value] > self.high_dist_threshold).astype(int)

        df[fenum.LOG_TRIP_DURATION.value]=np.log(df[fenum.TRIP_DURATION.value]) 


        if beast_feats: 
            df = df[fenum.best_features()]

        if drop_feats: 
            feats = fenum.get_drop_feats() 
            df.drop(labels=feats, axis=1, inplace=True)

        if add_landmarkb: 
            df = self.__add_landmark_features(df)

        if drop_outliayer: 
            self.outliayer_limts = self.__comput_outliayer_limts(df) 
            df = self.__apply_outliayer_limts(df) 

        return df, self.lable_encoder_season, self.lable_encoder_store, self.high_dist_threshold, self.outliayer_limts        



    def transform(self, df:pd.DataFrame, lable_encoder_season,  lable_encoder_store, hdt, outliayerLimits): 
            if outliayerLimits is not None: 
                self.outliayer_limts = outliayerLimits 

            df = df.copy() 
    
            df[fenum.PICKUP_DATETIME.value] = pd.to_datetime(df[fenum.PICKUP_DATETIME.value]) 
            df[fenum.HOUR.value] = df[fenum.PICKUP_DATETIME.value].dt.hour 
            df[fenum.MONTH.value] = df[fenum.PICKUP_DATETIME.value].dt.month 
            df[fenum.DAY_OF_WEEK.value] = df[fenum.PICKUP_DATETIME.value].dt.dayofweek 
    
            
            def getseason(month):
                if  4 <= month <= 7: return 'Sprint'
                elif  8 <= month <= 10: return 'Summar'
                elif  11 <= month <= 12: return 'Fail'
                else: return 'Winter ' 
    
    
            df[fenum.SEASON.value] = df[fenum.MONTH.value].apply(getseason)   
    
        
            df[fenum.SEASON.value] = lable_encoder_season.transform(df[fenum.SEASON.value]) 
    
         
            df[fenum.STORE_AND_FWD_FLAG.value] = lable_encoder_store.transform(df[fenum.STORE_AND_FWD_FLAG.value])
    
    
            df[fenum.IS_WEEKEND.value] = df[fenum.DAY_OF_WEEK.value].apply(lambda x: 1 if x >= 5 else 0)
    
            # 1. Flag late night hours
            df[fenum.IS_LATE_NIGHT.value] = df[fenum.HOUR.value].isin([20, 21, 22, 23, 0, 1, 2, 3, 4]).astype(int)
    
    
            # Binary flag for group vs. solo rides
            df[fenum.IS_GROUP_TRIP.value] = (df[fenum.PASSENGER_COUNT.value] > 1).astype(int) 
    
            df[fenum.IS_RUSH_HOUR.value] = df[fenum.HOUR.value].isin([7, 8, 9, 16, 17, 18]).astype(int)
    
            df[fenum.LOG_DISTANCE.value] = np.log1p(self.__calc_haversine_distance(
                lat1=df[fenum.PICKUP_LATITUDE.value], 
                lon1=df[fenum.PICKUP_LONGITUDE.value], 
                lat2=df[fenum.DROPOFF_LATITUDE.value], 
                lon2=df[fenum.DROPOFF_LONGITUDE.value] 
            )) 
    
            # Highlight long-haul trips (top 15% distance threshold)
            df[fenum.IS_LONG_HAUL.value] = (df[fenum.LOG_DISTANCE.value] > hdt).astype(int)


            df[fenum.LOG_TRIP_DURATION.value]=np.log(df[fenum.TRIP_DURATION.value]) 
                
            if beast_feats: 
                df = df[fenum.best_features()]
    
            if drop_feats: 
                feats = fenum.get_drop_feats() 
                df.drop(labels=feats, axis=1, inplace=True)
    
            if add_landmarkb: 
                df = self.__add_landmark_features(df)
    
            if drop_outliayer and self.outliayer_limts: 
                df = self.__apply_outliayer_limts(df) 
    
            return df 




    def polynomial_feature(self, x, x_val=None):
        self.poly = PolynomialFeatures(degree=poly_degree, include_bias=poly_include_bias)
        x_poly = self.poly.fit_transform(x) 
        if x_val is not None: 
            x_val_poly = self.poly.transform(x_val)  
            return self.poly, x_poly, x_val_poly 
        return self.poly, x_poly

    def scaling(self, x, x_val=None): 
        if scaling_option == 1: 
            self.scaler = MinMaxScaler()
        elif scaling_option == 2: 
            self.scaler = StandardScaler()
        elif scaling_option == 3: 
            self.scaler = Normalizer()
        else: 
            return None, x, x_val  

        x_scaled = self.scaler.fit_transform(x) 
        if x_val is not None: 
            x_val_scaled = self.scaler.transform(x_val) 
            return self.scaler, x_scaled, x_val_scaled 

        return self.scaler, x_scaled
                    
        



if __name__ == '__main__': 
    df = load_data(train_path) 
    print(df.shape)  
    process_pipline = Preprocessing_Pipline() 
    df, _, _ = process_pipline.fit_transform(df)
    target = fenum.LOG_TRIP_DURATION.value 
    x = df.drop(target, axis=1 , errors='ignore') 
    t = df[target] 
    print(x.shape)
    print(t.shape) 
    print(x.columns)

    










