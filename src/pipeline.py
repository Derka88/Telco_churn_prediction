from data_load import load_data
from features import preprocess_data
from train import train_model
from evaluate_model import evaluate_model
import joblib
import logging

logger = logging.getLogger(__name__)

def run_pipeline():

    logger.info('Start pipeline')

    df = load_data()

    # 2 features
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3 train
    model = train_model(X_train, y_train)

    # 4 evaluate
    evaluate_model(model, X_test, y_test)

    # 5 save
    joblib.dump(model, "model.pkl")

    logger.info('End pipeline')

if __name__ == "__main__":
    run_pipeline()