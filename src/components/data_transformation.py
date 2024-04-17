import os  # Importing the os module for operating system functionality
import sys  # Importing the sys module for system-specific parameters and functions
from src.exception import CustomException  # Importing CustomException from src.exception module
from src.logger import logging  # Importing logging from src.logger module
import pandas as pd  # Importing pandas library with an alias pd
import numpy as np  # Importing numpy library with an alias np
from sklearn.compose import ColumnTransformer  # Importing ColumnTransformer from sklearn.compose module
from sklearn.impute import SimpleImputer  # Importing SimpleImputer from sklearn.impute module
from sklearn.pipeline import Pipeline  # Importing Pipeline from sklearn.pipeline module
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Importing OneHotEncoder and StandardScaler from sklearn.preprocessing module
from dataclasses import dataclass  # Importing dataclass decorator from dataclasses module
from src.utils import save_object  # Importing save_object function from src.utils module

@dataclass  # Decorator indicating that the following class is a data class
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # File path for storing preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  # Initializing data_transformation_config with an instance of DataTransformationConfig

    def get_data_transformer_object(self):
        """
        Method to get the data transformer object which preprocesses numerical and categorical columns.

        Returns:
            ColumnTransformer: Preprocessor object for transforming numerical and categorical columns.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
            
            # Define pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with median
                    ("scaler", StandardScaler())  # Standardize numerical features
                ]
            )

            # Define pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # One-hot encode categorical features
                    ("scaler", StandardScaler(with_mean=False))  # Standardize categorical features
                ]
            )

            # Log information about numerical and categorical columns
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create ColumnTransformer to apply different transformations to numerical and categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply num_pipeline to numerical columns
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Apply cat_pipeline to categorical columns
                ]
            )

            return preprocessor  # Return the preprocessor object
        except Exception as e:
            raise CustomException(e, sys)  # Raise CustomException if an error occurs

    def initiate_data_transformation(self, train_path, test_path):
        """
        Method to initiate data transformation process.

        Args:
            train_path (str): Path to the training data file.
            test_path (str): Path to the test data file.

        Returns:
            tuple: A tuple containing transformed training and test data arrays, and the path to the preprocessor object file.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error(f"Error occurred during data transformation: {e}")
            raise CustomException(e, sys)  # Raise CustomException if an error occurs
