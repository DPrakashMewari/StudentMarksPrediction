import os
import sys
from dataclasses import dataclass
from src.exception import MyException
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.utils import save_object,evaluate_models


@dataclass
class ModeltrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModeltrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test = (train_arr[:,:-1],
                            train_arr[:,-1],
                            test_arr[:,:-1],
                            test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbours Regressor" : KNeighborsRegressor(),
                "XGBRegressor" : XGBRegressor(),
                "Adaboost Regressor" : AdaBoostRegressor()
            }

            # Applying Parameter for tuning All Models 
            params={
                "K-Neighbours Regressor":{},
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Adaboost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict= evaluate_models(X_train=X_train,y_train = y_train,X_test=X_test,y_test=y_test,
                                              models=models,param=params)
            # To Get Best Model Score from Dict
            best_model_score = max(sorted(model_report.values()))

            # To Get Best Model name from Dict 
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise MyException("No Best Model Found")
            
            logging.info("Found best model")
            save_object(filepath=self.model_trainer_config.trained_model_file_path,
                        obj=best_model)
            predicted = best_model.predict(X_test)
            r2_scores = r2_score(y_test,predicted)
            return r2_scores
        
        except Exception as e:
            raise MyException(e,sys)
    

