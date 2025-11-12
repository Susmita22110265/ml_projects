 # Data Transformation : This file prepares our data (the CSV from data ingestion) and turns it into clean, numeric, ready-to-use data for our machine learning model.

import sys # sys to work with files and folders
import numpy as np # To do array and math operations
import pandas as pd   # To create data frames and manipulate data
import os

from dataclasses import dataclass  # to use class in python for storing a particular config
from sklearn.compose import ColumnTransformer # 
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig : 
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation : 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transforemer(self):
        try :
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']

            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"numerical columns : {numerical_columns}")


            num_pipeline = Pipeline(
                steps = [('imputer', SimpleImputer(strategy = 'Median')),
                         ('scaler', StandardScaler())]
                         )

            cat_pipeline =  Pipeline(
                steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), 
                         ('onehotencoder', OneHotEncoder()), 
                         ('scaler', StandardScaler())]
                         )
            

            logging.info('Categorical Columns encoding completed')
            logging.info('Numerical Columns standard scalling completed')    
        
            preprocessor = ColumnTransformer[('numerical pipeline', num_pipeline, numerical_columns),
                                         ('catergorical pipeline', cat_pipeline, categorical_columns) ]


            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"numerical columns : {numerical_columns}")

            return preprocessor
    

        except Exception as e:
              raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path) :  
        try : 
           train_df = pd.read_csv(train_path)
           test_df = pd.read_csv(test_path)

           logging.info("Read train and test data completed")

           logging.info("Obtaining preprocessing object")

           preprocessing_object = self.get_data_transformer_object()

           target_column_name = "math score"
           numerical_columns = ['writing score', 'reading score']

           input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
           target_feature_train_df = train_df[target_column_name]


           input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
           target_feature_test_df = test_df[target_column_name]
           
           logging.info(f"Applying preprocessesing object on training dataframe and testing dataframe.")

           input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
           input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)
           
           train_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
         
           test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

           logging.info(f"Savedd preprocessing object. ")

           save_object(
               file_path = self.data_tranform_config.preprocessor_obj_file_path, 
               obj = preprocessing_object
           )


           return (
               train_arr, 
               test_arr, 
               self.data_transformation_config.preprocessor_obj_file_path
           )
        
        
        except Exception as e : 
            raise CustomException(e,sys) 




