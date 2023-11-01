# First thing is to read the data from whatever data source it comes from
# We will split the data into training and testing

import os
import sys #because we will be using our CustomException and logging the errors
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


# there will be some sort of inputs to the data ingestion process, like where should raw data be stored, or where should testing data be stored, what path, etc.
# for that we will create another class

@dataclass # this decorator makes the class initialization very simple and saves the creation of __init__, __repr__ and __eq__ methods to make the code more readable.
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("../../notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe") # important to keep writing logs

            # it's gonna create a directory in the path of training data path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)



# The reason for this structure is to ensure that the data ingestion process is only executed when you run the script directly. 
# If you were to import this script as a module into another script, the data ingestion process wouldn't start automatically. 
# This separation allows you to reuse the DataIngestion class and its methods in other scripts without running the data ingestion process unless explicitly invoked.
if __name__ == "__main__":
    # Instantiate DataIngestion() class
    obj = DataIngestion()
    # Split the train and test data. Save them in a .csv format in artifacts folder
    train_data,test_data = obj.initiate_data_ingestion()
    # Instantiate DataTransformation() class
    data_transformation = DataTransformation()
    # Transform the data and save the preprocessor object in a pickle file in artifacts folder
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)
    # Instantiate the ModelTrainer() class
    modeltrainer = ModelTrainer()
    # Input the training and testing arrays to the initiate_model_training function to get the r2 score
    print(modeltrainer.initiate_model_training(train_arr, test_arr))