import os  # Importing the os module for operating system functionality
import sys  # Importing the sys module for system-specific parameters and functions
from src.exception import CustomException  # Importing CustomException from src.exception module
from src.logger import logging  # Importing logging from src.logger module
import pandas as pd  # Importing pandas library with an alias pd
from sklearn.model_selection import train_test_split  # Importing train_test_split function from sklearn.model_selection module
from dataclasses import dataclass  # Importing dataclass decorator from dataclasses module
from src.components.data_transformation import DataTransformation, DataTransformationConfig


@dataclass  # Decorator indicating that the following class is a data class
class DataIngestionConfig:  # Defining a data class named DataIngestionConfig
    train_data_path: str=os.path.join('artifacts','train.csv')  # Default path for training data
    test_data_path: str=os.path.join('artifacts','test.csv')  # Default path for testing data
    raw_data_path: str=os.path.join('artifacts','data.csv')  # Default path for raw data

class DataIngestion:  # Defining a class named DataIngestion
    def __init__(self):  # Constructor method to initialize class instances
        self.ingestion_config = DataIngestionConfig()  # Initializing ingestion_config with an instance of DataIngestionConfig
    
    def initiate_data_ingestion(self):  # Method to initiate data ingestion process
        logging.info("Entered the data ingestion method or component")  # Logging information about entering the data ingestion method
        try:  # Starting a try block to handle potential exceptions
            df = pd.read_csv('notebook/data/stud.csv')  # Reading the dataset as a pandas DataFrame
            logging.info('Read the dataset as dataframe')  # Logging information about successfully reading the dataset

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  # Creating directories if they don't exist
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)  # Saving the raw data to the specified path

            logging.info("Train test split initiated")  # Logging information about initiating train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)  # Splitting the data into training and testing sets

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)  # Saving the training data to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)  # Saving the testing data to the specified path

            logging.info("Ingestion of the data is completed")  # Logging information about completing the data ingestion process
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)  # Returning the paths of the training and testing data
        except Exception as e:  # Handling any exception that occurs
            raise CustomException(e, sys)  # Raising a CustomException with the caught exception and the sys module

if __name__ == "__main__":  # Entry point of the script
    obj = DataIngestion()  # Creating an instance of DataIngestion class
    train_data,test_data=obj.initiate_data_ingestion()  # Calling the method to initiate data ingestion process
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)