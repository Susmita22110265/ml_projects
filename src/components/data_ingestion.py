# data ingestion : Collecting raw data and preparing it for our project 

import os # helps us work with files and folders 
import pandas as pd # helps us analyze the data using dataFrames 
from src.exception import CustomException  # Nice way to tell about exceptions such file not found, etc. 
from src.logger import logging # Journaling every process, responsible for showing messages like : Loading data, Training started, etc. 
from sklearn.model_selection import train_test_split  # Used to split the data into training and testing set. 
from dataclasses import dataclass # creates a class in python for storing configurations 
import sys # To understand error better, gives all information about error

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:  # Class which helps us store our training data file, testing data file and original data file for safety.
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path :str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'data.csv')
# All the file paths are saved.

class DataIngestion :     # class that helps us do this saving files 
    def __init__(self):   # helps us create an object that has 3 varaibles that will store the path of the files. 
        self.ingestion_config = DataIngestionConfig()  # object created
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try : 
            df = pd.read_csv('C:/Users/kavit/OneDrive/Desktop/ml_projects/notebook/StudentsPerformance.csv') # Reading the raw data and storing it as a dataframe
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True) # creating a folder called artifacts. If it already exists, then don't create and crash the folder
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True) # store the entire dataframe as raw data for safety in the artifacts folder

            logging.info("Train Test split initiated")
            
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42) # Train Test split
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True) # store the training set to artifacts folder 

            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True) # store the testing set to the artifacts folder 

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path,
            )                                               # returning training and testing data path for further processes
        
        except Exception as e:
            raise CustomException(e,sys) 



# So when this data ingestion file is called, we will return the training and testing dataset for data transformation, EDA and model training 

if __name__ == "__main__" : 
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
    



# To run this file use the command ->    python -m src.components.data_ingestion


# this makes sure that the project structure and other internal runnings are taken care of