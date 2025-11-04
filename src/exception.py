# helps us get the information about error other than the error message.  
import sys 
import logging

# if an error is occuring in our project , we pass the error message and error detail to the below function and use sys to get more info such as error filename, linenumber, etc. 
def error_message_detail(error, error_detail:sys):
    # getting the error details (type, message, traceback) using the function exc_info
    # traceback exc.tb variable is very important as it contains information related to filename, error line number and error message

    _,_,exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename  # getting the file name in which the error occured
    line_number = exc_tb.tb_lineno  # getting the line number in which the error occured in the file

    # printing the error details and the error 
    error_message = f'Error occured in the file[{file_name}] and the line number is [{line_number}] and Error message is [{str(error)}]'

    return error_message



class CustomException(Exception): 
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message

