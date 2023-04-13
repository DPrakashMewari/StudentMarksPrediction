import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import MyException
from src.logger import logging
import os 
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_obj_file_path:str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    """ This is Used For Data Transformation """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise MyException(e,sys)
    def initiate_data_transformation(self,train,test):
        """Applying Here Transformation And Dump into Pickle In Short"""
        try:
            train_df = pd.read_csv(train)
            test_df = pd.read_csv(test)
            logging.info("Read train and test data Completed ")
            logging.info("Obtaining preprocessing Object")
            preprocessing = self.get_data_transformer_object()
            target_coloumn = "math_score"
            # numerical_columns = ['writing','reading_score']
                        
            input_feature_train_df=train_df.drop(columns=[target_coloumn],axis=1)
            target_feature_train_df=train_df[target_coloumn]

            input_feature_test_df=test_df.drop(columns=[target_coloumn],axis=1)
            target_feature_test_df=test_df[target_coloumn]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr = preprocessing.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"Saved preprocessing object.")

            save_object(
                filepath = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing
            )

            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise MyException(e,sys)