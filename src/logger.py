import logging  # Importing the logging module for logging functionality
import os  # Importing the os module for operating system functionality
from datetime import datetime  # Importing datetime module to work with dates and times

# Generating a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Creating the directory path for storing log files
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Creating the directory if it doesn't exist to store log files
os.makedirs(logs_path, exist_ok=True)

# Creating the full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configuring the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Specifying the log file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Specifying the log message format
    level=logging.INFO,  # Setting the logging level to INFO
)
