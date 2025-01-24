import pandas as pd
import mlflow
import pickle
from sklearn.metrics import accuracy_score
import os
import yaml
from urllib.parse import urlparse






params=yaml.safe_load(open('params.yaml'))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    x=data.drop(columns="Outcome")
    y=data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/garvkhurana/firstmlopspipeline.mlflow")
    
    
    model=pickle.load(open(model_path,"rb"))
    
    predictions=model.predict(x)
    accuracy=accuracy_score(y,predictions)
    
    
    mlflow.log_metric("accuracy",accuracy)
    print(f"model accuracy {accuracy}")
    
    
    if __name__=="__main__":
        evaluate(data_path=params["data_path"],model_path=params["model_path"])
