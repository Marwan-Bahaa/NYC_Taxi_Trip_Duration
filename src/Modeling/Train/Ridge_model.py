import sys
from pathlib import Path

# Set project root (NYC_Taxi_Trip_Duration) and src directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

# Add both to sys.path
for path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from sklearn.linear_model import Ridge 
from Config.load import load_config

# FIX: Import from Modeling.Helper instead of Helper
from Modeling.Helper.eval import evaluate_model  
from log.apply_log import log_result
from Modeling.Helper.prepare import Preparing
from Modeling.Helper.save import save_model


config = load_config() 

alpha = config['model']['alpha'] 
fit_intercept = config['model']['fit_intercept']


class Train():
    def __init__(self, x_train, x_val, t_train, t_val):
        self.x_train = x_train
        self.x_val = x_val
        self.t_train = t_train
        self.t_val = t_val

    def try_ridge(self): 
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept) 
        model.fit(X=self.x_train, y=self.t_train)
        log_result(f'Train Path', 'Ridge')
        train_error, train_score = evaluate_model(model, self.x_train, self.t_train, 'train')
        val_error, val_score = evaluate_model(model, self.x_val, self.t_val, 'val')


        log_result(f"fit-transform: {fit_intercept}", "Ridge")
        log_result(f"MSE for Train: {train_error}", "Ridge")
        log_result(f"R2Score for Train: {train_score}", "Ridge")
        log_result(f"MSE for Val: {val_error}", "Ridge")
        log_result(f"R2Score for Val: {val_score}", "Ridge")
        log_result('--'*40, "Ridge")

        return model 


if __name__ == '__main__':
    prepare = Preparing()
    x_train, x_val,  t_train, t_val, encode_season, encode_store, poly, scaler = prepare.prepare_data()
    print(x_train.shape, t_train.shape, x_val.shape, t_val.shape)    

    train = Train(x_train, x_val, t_train, t_val)
    model = train.try_ridge()
    evaluate_model(model, x_train, t_train, 'train')
    evaluate_model(model, x_val, t_val, 'val')

    print("Successful")
    save_model(model=model, 
               encode_season=encode_season,
               encode_store=encode_store, 
               poly=poly,
               scaler=scaler, 
               name="Ridge",
               tyep='val') 
    
    