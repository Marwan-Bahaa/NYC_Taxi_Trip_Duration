from Helper import Preparing 
from Train.Ridge_model import Train 
from Helper.save import save_model


if __name__ == '__main__':
    prepare = Preparing()
    # Capture outliayer_limts from prepare_data
    x_train, x_val, t_train, t_val, enc_season, enc_store, poly, scaler, hdt, limits = prepare.prepare_data()
    
    train = Train(x_train, x_val, t_train, t_val)
    model = train.try_ridge()

    feature_names = x_train.columns.tolist() if hasattr(x_train, 'columns') else None
    save_model(model, enc_season, enc_store, poly, scaler, hdt, limits, feature_names)