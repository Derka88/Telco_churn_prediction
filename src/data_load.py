import pandas as pd
import os

def load_data() : 

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    chemin = os.path.join(base_dir, '..', 'data', 'data.csv')
    
    df = pd.read_csv(chemin)

    return df