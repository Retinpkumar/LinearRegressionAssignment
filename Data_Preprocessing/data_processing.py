import pandas as pd
import numpy as np
# from scipy import stats
from application_logger.app_logger import Logger
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    This class is used for cleaning the raw data before it is fed to the model.
    Written by: Retin P Kumar
    """
    def __init__(self, df):
        self.df = df
        self.logger_object = Logger()
        self.logfile_path = 'LogFiles/preprocessing_log.txt'
        self.logfile = open(self.logfile_path, mode='a')

    def cleanData(self):
        """
        Method name: cleanData
        Description: This method is used for cleaning the raw data.
        Output: Pandas dataframe
        On failure: Raise Exception
        """
        self.logfile = open(self.logfile_path, mode='a')
        self.logger_object.log(self.logfile, "Accessing the method 'cleanData' from class 'DataProcessor'")
        try:
            self.logger_object.log(self.logfile, "Data loaded successfully as a dataframe.")
            self.logfile.close()
        except Exception as e:
            self.logger_object.log(self.logfile, "Exception occured while loading dataframe. Exception :" + str(e))
            self.logger_object.log(self.logfile, "Data loading unsuccessful. Exited the method get_trainingData")
            self.logfile.close()
            raise Exception()

        try:
            self.logfile = open(self.logfile_path, mode='a')
            #log transformation
            for col in ['NOX','DIS','LSTAT']:
                    self.df[col] = np.log1p(self.df[col])
            self.logger_object.log(self.logfile, "Log transformation performed successfully.")
            self.logfile.close()
        except Exception as e:
            self.logger_object.log(self.logfile, "Exception while performing log transformation. Exception :" + str(e))
            self.logger_object.log(self.logfile, "Log transformation unsuccessful. Exited the method get_trainingData")
            self.logfile.close()
            raise Exception()
        
        try:
            self.logfile = open(self.logfile_path, mode='a')
            #Power transformation
            for col in ['CRIM', 'ZN']:
                    self.df[col] = np.power(self.df[col], 0.03125)
            self.logger_object.log(self.logfile, "Power transformation performed successfully.")
            self.logfile.close()
        except Exception as e:
            self.logger_object.log(self.logfile, "Exception while performing power transformation. Exception :" + str(e))
            self.logger_object.log(self.logfile, "Power transformation unsuccessful. Exited the method get_trainingData")
            self.logfile.close()
            raise Exception()
                       
        return self.df