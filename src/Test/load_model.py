import joblib 

# load_model.py
def load_model_ridge(name="Ridge", type="val"):
    filename = f'val_pkl/{name}.pkl' if type == "val" else f'{name}.pkl'
    d = joblib.load(filename)
    return d["model"], d["encode_season"], d["encode_store"], d["poly"], d["scaler"], d["hdt"], d["outliayer_limts"], d["feature_names"] 
