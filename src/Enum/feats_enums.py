from enum import Enum
from typing import List

class FeatureEnum(str, Enum):
    ID = "id"
    VENDOR_ID = "vendor_id"
    PICKUP_DATETIME = "pickup_datetime"
    PICKUP_LONGITUDE = "pickup_longitude"
    PICKUP_LATITUDE = "pickup_latitude"
    PASSENGER_COUNT = "passenger_count"
    DROPOFF_LONGITUDE = "dropoff_longitude"
    DROPOFF_LATITUDE = "dropoff_latitude"
    STORE_AND_FWD_FLAG = "store_and_fwd_flag"
    HAVERSINE_DISTANCE = "haversine_distance"

    TRIP_DURATION = "trip_duration"
    HOUR = "hour"
    YEAR = "year"
    DAY_OF_WEEK = "day_of_week"
    MONTH = "month"
    SEASON = "season"
    
    # Target Variable
    LOG_TRIP_DURATION = "log_trip_duration"

    # Spatial & Distance Features
    LOG_DISTANCE = "log_distance"
    PICKUP_DIST_TO_CENTER = "pickup_dist_to_center"
    DROPOFF_DIST_TO_CENTER = "dropoff_dist_to_center"
    PICKUP_DIST_JFK = "pickup_dist_jfk"
    DROPOFF_DIST_JFK = "dropoff_dist_jfk"
    PICKUP_DIST_LGA = "pickup_dist_lga"
    DROPOFF_DIST_LGA = "dropoff_dist_lga"
    PICKUP_DIST_EWR = "pickup_dist_ewr"
    DROPOFF_DIST_EWR = "dropoff_dist_ewr"
    PICKUP_DIST_TIMES_SQUARE = "pickup_dist_times_square"
    DROPOFF_DIST_TIMES_SQUARE = "dropoff_dist_times_square"

    # Flags & Indicators
    IS_LONG_HAUL = "is_long_haul"
    IS_WEEKEND = "is_weekend"
    IS_LATE_NIGHT = "is_late_night"
    IS_GROUP_TRIP = "is_group_trip"
    WEEKEND_NIGHT_GROUP = "weekend_night_group"
    IS_RUSH_HOUR = "is_rush_hour"
    IS_AIRPORT_JFK = "is_airport_jfk"
    IS_AIRPORT_LGA = "is_airport_lga"
    IS_AIRPORT_EWR = "is_airport_ewr"
    IS_ANY_AIRPORT = "is_any_airport"
    VENDOR_ID_2 = "vendor_id_2"

    # One-Hot Encoded Seasons
    SEASON_SPRING = "season_Spring"
    SEASON_SUMMER = "season_Summer"
    SEASON_WINTER = "season_Winter"

    # Cyclical Time Features
    HOUR_SIN = "hour_sin"
    HOUR_COS = "hour_cos"
    DOW_SIN = "dow_sin"
    DOW_COS = "dow_cos"
    MONTH_SIN = "month_sin"
    MONTH_COS = "month_cos"





    @classmethod
    def get_all_features(cls) -> List[str]:
        """Returns all 36 feature names for X (excluding target)."""
        return [member.value for member in cls if member != cls.LOG_TRIP_DURATION]

    @classmethod
    def best_features(cls) -> List[str]:
        """Top spatial & distance features subset."""
        return [
            cls.LOG_DISTANCE,
            cls.DROPOFF_LONGITUDE,
            cls.PICKUP_LONGITUDE,
            cls.DROPOFF_LATITUDE,
            cls.PICKUP_LATITUDE,
        ]

    @classmethod 
    def get_drop_feats(cls) -> List[str]:
        return[
            cls.PICKUP_DATETIME, 
            cls.ID, 
            cls.TRIP_DURATION
        ] 
