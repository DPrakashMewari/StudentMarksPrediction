import os 
import sys
from sklearn.metrics import r2_score
import pickle
from src.exception import MyException

def save_object(filepath,obj):
    try:
       dir_path = os.path.dirname(filepath)
       os.makedirs(dir_path,exist_ok=True)
       with open(filepath,'wb') as fo:
            pickle.dump(obj,fo)

    except Exception as e:
        raise MyException(e,sys)    
    
def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]


            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        raise MyException (e,sys)