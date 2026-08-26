import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.metrics import r2_score, mean_squared_error 



def evaluate_model(model, x, t, name='val'): 
       t_pre = model.predict(x) 
       mse = mean_squared_error(y_true=t, y_pred=t_pre) 
       r2 = r2_score(y_true=t, y_pred=t_pre) 
       print(f'{name} Evaluation:\n MSE : {mse} \t R2Score {r2}')
       return mse, r2 
 


