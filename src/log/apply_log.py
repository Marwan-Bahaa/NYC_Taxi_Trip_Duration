import os 

def log_result(text, name='ridge', type='train', filename=None): 
    if filename is None: 
        if type == 'test': 
            filename = fr'src/log/test_results/test_model{name}' 

        else: 
            filename = fr'src/log/results/model_results_{name}.txt' 

    os.makedirs(os.path.dirname(filename), exist_ok=True)  

    with open(filename, 'a') as f: 
        f.write(text) 
        f.write('\n') 


                    