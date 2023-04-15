import os 
import sys
from src.exception import MyException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModeltrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path : str =os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('Entered in the data ingestion')
        try:
            # Read from Folder
            df = pd.read_csv('./notebook/data/stud.csv')
            logging.info('Readed the dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Data Dump
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split is started")

            # Splitting data 
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            # Dump Train
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            # Dump Test
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')

            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
        except Exception as e:
            raise MyException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train,test=obj.initiate_data_ingestion()
    data_tranformation = DataTransformation()
    train_s,test_s,path = data_tranformation.initiate_data_transformation(train,test)
    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_s,test_s))