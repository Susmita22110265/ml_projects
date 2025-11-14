import sys
import os
import numpy as np
import pandas as pd


from dataclasses import dataclass
#from catboost import CatBoostRegressor

from sklearn.ensemble import(
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

class ModelTrainerConfig :
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')  # we will store the trained model file path in a variable in this class

class ModelTrainer : 
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()  # Initiating a model trainer configuration instance

    def initiate_model_trainer(self, train_array, test_array) : 
        try : 
            logging.info("Splitting training and test input data")    
# We will now split the training and testing data. Later we will split the training data into X and y. With X being the data trained on to predict y
            X_train , y_train, X_test, y_test = (train_array[:, : -1], 
                                                 train_array[:, -1], 
                                                 test_array[:, : -1], 
                                                 test_array[:, -1])
           
            models = {
                "Random Forest " : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(), 
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(), 
                "K - Neighbors Classifier" : KNeighborsRegressor(), 
                "XGBClassifier " : XGBRegressor(),
                #"CatBoosting Classifier" : CatBoostRegressor(verbose = False),
                "AdaBoost Classifier" : AdaBoostRegressor()
            }

            model_report : dict = evaluate_model(X_train = X_train,    # evaluate model function will fit on the training data and will predic the y. r2_score is calculated based on the accuracy of how good y_prediction aligns with y_original 
                                                 y_train = y_train,
                                                 X_test = X_test, 
                                                 y_test = y_test, 
                                                 models = models)
            
            # First train on X_train and y_train to learn the patterns
            # Input X_test to predict y_test
            # Compute r2_score of y_test and y_test_prediction

            # In the below code we will save the model name that gives the best r2_score
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model foiunf")

            # We will store the trained model in the form of .pkl file that gives the highest r2_score
            save_object(file_path = self.model_trainer.trained_model_file_path, 
                        obj = best_model )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square  # Return the best r2_score. 

        except Exception as e:
            raise CustomException(e,sys)



    


