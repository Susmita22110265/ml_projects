 # Data Transformation : This file prepares our data (the CSV from data ingestion) and turns it into clean, numeric, ready-to-use data for our machine learning model.

import sys # sys to work with files and folders
import numpy as np # To do array and math operations
import pandas as pd   # To create data frames and manipulate data
import os # To work with files and folders

from dataclasses import dataclass  # to use class in python for storing a particular config
from sklearn.compose import ColumnTransformer # Transforms a column 
from sklearn.impute import SimpleImputer  # Imputes missing values with valid data such as Mode, Median ,etc. 
from sklearn.pipeline import Pipeline  # Used a create a pipeline of operations that is to be done on a dataframe in sequence
from sklearn.preprocessing import StandardScaler, OneHotEncoder # For standardizing the numerical features with mean 0, std_dev =1, and convert all categorical columns to machine readable form. 

from src.exception import CustomException # Displaying exceptions while running any code file
from src.logger import logging # Jounaling the journey 
from src.utils import save_object  # Saves the preprocessor into a file (.pkl format) so it can be reused



@dataclass
class DataTransformationConfig : # Creates a variable that stores the pkl file in the artifacts folder
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation : 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # create an object of the class where you can store the .pkl file that you will be creating 

    def get_data_transformer(self):  # return preprocessor object that has 2 pipelines that can transform data
        try :
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']

            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"numerical columns : {numerical_columns}")

            # 1st pipeline is a pipeline to transform numerical features
            num_pipeline = Pipeline(
                steps = [('imputer', SimpleImputer(strategy = 'median')),
                         ('scaler', StandardScaler())]
                         )
            
            # 2nd pipeline is a pipeline to transform categorical features
            cat_pipeline =  Pipeline(
                steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), 
                         ('onehotencoder', OneHotEncoder()), 
                         ('scaler', StandardScaler(with_mean=False))]
                         )
            

            logging.info('Categorical Columns encoding completed')
            logging.info('Numerical Columns standard scalling completed')    
        
            # preprocessor is an object that contains both the pipelines and when called transforms the entire dataframe according to their type
            preprocessor = ColumnTransformer([('numerical pipeline', num_pipeline, numerical_columns),
                                         ('catergorical pipeline', cat_pipeline, categorical_columns)])


            logging.info(f"categorical columns : {categorical_columns}")
            logging.info(f"numerical columns : {numerical_columns}")

            return preprocessor
    

        except Exception as e:
              raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path) :  
        # this function applies those transformations to train and test data.
        
        try : 
           # Reading the datasets
           train_df = pd.read_csv(train_path)
           test_df = pd.read_csv(test_path)

           logging.info("Read train and test data completed")

           logging.info("Obtaining preprocessing object")

           preprocessing_object = self.get_data_transformer()  # get an instance from the class DataTransformation's method get_data_traansformer()

           target_column_name = "math score"  # Feature to be predicted
           numerical_columns = ['writing score', 'reading score'] 

           input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)  # Features other than math score column in training dataset
           target_feature_train_df = train_df[target_column_name]   # Target feature (math_score) in training dataset


           input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1) # Features other than math_score column in the testing dataset
           target_feature_test_df = test_df[target_column_name]  # Target feature (math_score) in testing dataset
           
           logging.info(f"Applying preprocessesing object on training dataframe and testing dataframe.")

           input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)  # Applying preprocessing object using .fit_transform for training dataset
           input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)  # Applying preprocessing object using .transform for testing dataset
           
           # Note : fit_transform is used to learn and apply the transformation. So it is done on the training dataset only 
           # tranform() is only used to apply column transformation. Hence it is done on testing dataset

           train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # Concatenates transformed X_train and Y_train into a numpy 2D array
         
           test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]  # Concatenates transformed X_test and Y_test into a numpy 2D array

           logging.info(f"Saved preprocessing object. ")

           # we save the .pkl file in the artifacts folder which contains the Python object 
           save_object(
               file_path = self.data_transformation_config.preprocessor_obj_file_path, 
               obj = preprocessing_object
           )


           return (
               train_arr, 
               test_arr, 
               self.data_transformation_config.preprocessor_obj_file_path
           )
        
        
        except Exception as e : 
            raise CustomException(e,sys) 




