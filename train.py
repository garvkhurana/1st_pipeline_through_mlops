import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import os
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import mlflow
from urllib.parse import urlparse



os.environ['MLFLOW_TRACKING_URL'] = "https://dagshub.com/garvkhurana/firstmlopspipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "garvkhurana"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "00ffbefccd789ae963b92c71cfff3114090e5647"


params=yaml.safe_load(open('params.yaml'))['train']

def hyperparameter_tuning(x_train,y_train,param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
    grid_search.fit(x_train,y_train)
    return grid_search

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path,header=0)
    x = data.drop(columns=["Outcome"])  
    y = data["Outcome"]                 
 
    
    mlflow.set_tracking_uri("https://dagshub.com/garvkhurana/firstmlopspipeline.mlflow")
    
    mlflow.set_experiment("firstmlopspipeline")
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=random_state)
    signature=infer_signature(x_train,y_train)
    
    ##define hyperparameter grid
    param_grid={
        "n_estimators":[100,200],
        "max_depth":[5,10,None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
      
      
    grid_search=hyperparameter_tuning(x_train,y_train,param_grid)  
    
    best_model=grid_search.best_estimator_
    
    y_pred=best_model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    
    
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_param('best_n_estimators',best_model.n_estimators)
    mlflow.log_param('best_max_depth',best_model.max_depth)
    mlflow.log_param('best_min_samples_split',best_model.min_samples_split)
    mlflow.log_param('best_min_samples_leaf',best_model.min_samples_leaf)
    
    cm=confusion_matrix(y_test,y_pred)
    cr=classification_report(y_test,y_pred)
    
    mlflow.log_text(str(cm),"confusion_matrix.txt")
    mlflow.log_text(str(cr),"classification_report.txt")
    
    
    tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
    
    if tracking_url_type_store!="file":
        mlflow.sklearn.log_model(best_model,model_path,signature=signature)
    
    else:
        mlflow.sklearn.log_model(best_model,"model",signature=signature)
        
        
        
        
        
    if not os.path.exists(os.path.dirname(model_path)) and os.path.dirname(model_path) != "":
      os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save the model using pickle
      with open(model_path, "wb") as f:
             pickle.dump(best_model, f)

    print("Model saved to", model_path)

    
if __name__=="__main__":
    train(params["data"],params["model"],params["random_state"],params["n_estimators"],params["max_depth"])     
    
    
