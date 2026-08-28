import joblib
# save.py
def save_model(model, encode_season, encode_store, poly, scaler, hdt, outliayer_limts, feature_names, name="Ridge", type='test'):
    model_dict = {
        "model": model,
        "encode_season": encode_season,
        "encode_store": encode_store,
        "poly": poly,
        "scaler": scaler, 
        "hdt": hdt,
        "outliayer_limts": outliayer_limts,
        "feature_names": feature_names
    }
    filename = f'val_pkl/{name}.pkl' if type == 'val' else f'{name}.pkl'
    joblib.dump(model_dict, filename)