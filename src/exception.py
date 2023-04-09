import sys
import logging 

def error_message_detail(error,error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occured in python Script name [{0}] line number [{2}] error message [{2}]".format(filename,exc_tb.tb_lineno,str(error))
    return error_message

class MyException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details=error_detail)

    def __str__(self):
        return self.error_message

if __name__ ==  "__main__":
    try : 
        a = 1/0 
    except Exception as e:
        logging.info("Error Happend")
        raise MyException(e,sys) 