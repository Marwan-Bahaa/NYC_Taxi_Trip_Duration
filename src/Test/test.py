import os
import sys

# 1. Add project root & src to sys.path BEFORE loading internal packages
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in [ROOT_DIR, SRC_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


from src.data.load_data import load_data 
from src.Test.load_model import load_model_ridge 
from Enum.path_enums import PathEnum 
from Enum.feats_enums import FeatureEnum as fenum
from Preprocessing import Preprocessing_Pipline 
from sklearn.metrics import mean_squared_error, r2_score
from src.Config.load import load_config

config = load_config()
test_path = PathEnum.TEST_PATH.value
name = config['model']['name'] 

def run_test():
    df = load_data(test_path)
    model, enc_season, enc_store, poly, scaler, hdt, limits, feature_names = load_model_ridge(name, "test")

    prepare = Preprocessing_Pipline()
    df = prepare.transform(df, enc_season, enc_store, hdt, outliayerLimits=limits)

    target_col = fenum.LOG_TRIP_DURATION.value
    t = df[target_col].copy()
    x = df.drop(columns=[target_col], errors="ignore")

    # Enforce exact column order from training
    if feature_names:
        x = x[feature_names]

    x_poly = poly.transform(x)
    x_scaled = scaler.transform(x_poly)
    pred = model.predict(x_scaled)

    mse_error = mean_squared_error(t, pred)
    r2score = r2_score(t, pred)
    print(f"r2score: {r2score:.4f}, mse: {mse_error:.4f}") 


if __name__ == '__main__': 
    run_test()