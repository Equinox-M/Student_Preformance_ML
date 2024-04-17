import sys  # Importing the sys module for system-specific parameters and functions
from src.logger import logging  # Importing logging from src.logger module

def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message including the file name, line number, and error message.
    
    Args:
        error: The error message.
        error_detail: System information related to the error.
        
    Returns:
        str: Detailed error message including file name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    # Constructing the error message with file name, line number, and error message
    error_message = 'Error occurred in Python script name [{0}] line number [{1}] error message [{2}]'.format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class inheriting from Exception class.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with the error message and detailed system information.

        Args:
            error_message: The error message.
            error_detail: System information related to the error.
        """
        super().__init__(error_message)
        # Creating a detailed error message using error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """
        Returns a string representation of the CustomException.

        Returns:
            str: String representation of the CustomException.
        """
        return self.error_message
