import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def split_train_test(df, var_cible) : 
    
    X = df.drop([var_cible], axis=1)
    y = df[var_cible]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.20, stratify=y)
    
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    
    return X_train, X_test, y_train, y_test

def recodage_var_explicative(df):
    
    cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    cols_drop = ['customerID','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','tenure','gender', 'Partner', 'PhoneService']

    df['score_services'] = df[cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

    df['anciennete'] = pd.cut(df['tenure'],
                        bins=[0, 20, 40, 60, 80],
                        labels=['0_20M', '21_40M', '41_60M', '61_80M'])
    

    df_model = df.drop(cols_drop, axis=1)

    return df_model

def encodage_var_cible(df) :
    
    le = LabelEncoder()
    df = le.fit_transform(df)

    return df

def encodage_var_explicative(df, preprocessor=None):
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if preprocessor is None:  # on est sur le train → on fit
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ])
        df_enc = preprocessor.fit_transform(df)
    else:  # on est sur le test → on transform seulement
        df_enc = preprocessor.transform(df)

    return df_enc, preprocessor