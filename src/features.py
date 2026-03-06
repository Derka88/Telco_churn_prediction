from utils import (split_train_test,recodage_var_explicative, 
                   encodage_var_cible,encodage_var_explicative)

def preprocess_data(df):
    
    X_train, X_test, y_train, y_test = split_train_test(df, 'Churn')

    X_train = recodage_var_explicative(X_train)

    X_test = recodage_var_explicative(X_test)

    X_train_enc, preprocessor = encodage_var_explicative(X_train)

    X_test_enc, _ = encodage_var_explicative(X_test, preprocessor)

    y_train_enc = encodage_var_cible(y_train)

    y_test_enc = encodage_var_cible(y_test)
     
    print(X_train_enc.shape,X_test_enc.shape,y_train_enc.shape,y_test_enc.shape)

    return X_train_enc, X_test_enc, y_train_enc, y_test_enc


