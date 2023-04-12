import os 
import sys

import pickle
from src.exception import MyException

def save_object(filepath,obj):
    try:
       dir_path = os.path.dirname(filepath)
       os.makedirs(dir_path,exist_ok=True)
       with open(filepath,'wb') as fo:
            pickle.dump(obj,fo)

    except Exception as e:
        raise MyException(e,sys)    
    