import os
import sys #because we will be using our CustomException and logging the errors
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.utils import save_object

@dataclass # this decorator makes the class initialization very simple and saves the creation of __init__, __repr__ and __eq__ methods to make the code more readable.
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    # raw_data_path: str=os.path.join("artifacts","data.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # inside data_transformation_config there will be the variables defined just in DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation. Creates a ColumnTransformer that contains the required transformations (defined as num_pipeline and cat_pipeline)
        applied to the required features of the dataset.
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            num_pipeline = Pipeline(

                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    # After applying ohe the data becomes very spare (many zeros) and centering the data can lead to erros. 
                    # Then, data is not centered (still scaled by dividing by the std).
                    ("scaler", StandardScaler(with_mean=False)) 
                ]

            )
            logging.info("Numericals columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function takes the train and test dataset paths. Uses the preprocessing_obj defined during get_data_transformer_object().
        Applies the preprocessing_obj to the training and testing arrays.
        Uses save_object() function defined in utils.py to save the preprocessing_obj after performing the transformations over the train and test arrays, 
        in a given path defined in DataTransformationConfig().
        
        The output of this function is the transformed train and test arrays, as well as the path of the transformation object (preprocessing_obj)
        
        '''

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)