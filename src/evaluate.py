import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Sebastian-LG/project-MLpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Sebastian-LG"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "13654272f67ed34439f32e197ce160c1d28e4b5f" # Token from DVC Dagshub remote


params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns = ['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    model = pickle.load(open(model_path,'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy: {accuracy}")


if __name__ == '__main__':
    evaluate(params['data'],params['model'])