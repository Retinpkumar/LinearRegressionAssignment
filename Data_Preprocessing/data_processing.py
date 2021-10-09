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
    def __init__(self):
        self.file = 'UserInput/test.csv'
        self.logger_object = Logger()
        self.logfile_path = 'LogFiles/preprocessing_log.txt'
        self.df = pd.read_csv(self.file, index_col=0)  # Creating a dataframe
        self.dtype_feat = ['RAD']
        self.boxcox_feat = ['CRIM', 'DIS', 'LSTAT']
        self.drop_cols = ['CHAS', 'NOX', 'PTRATIO']
        self.logfile = open(self.logfile_path, mode='a')
        self.scaler = StandardScaler()

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
            for feat in self.dtype_feat:  # Changing data types
                self.df[feat] = self.df[feat].astype('int')
                self.df[feat] = self.df[feat].astype('category')
            self.logger_object.log(self.logfile, "Data type conversion performed successfully.")
            self.logfile.close()
        except Exception as e:
            self.logger_object.log(self.logfile, "Exception while performing datatype conversion. Exception :" + str(e))
            self.logger_object.log(self.logfile, "Datatype conversion unsuccessful. Exited the method get_trainingData")
            self.logfile.close()
            raise Exception()

        try:
            self.logfile = open(self.logfile_path, mode='a')
            # Log/BoxCox transform
            for bfeat in self.boxcox_feat:
                self.df[bfeat] = np.log1p(self.df[bfeat])
            self.logger_object.log(self.logfile, "Successfully performed BoxCox transformation.")
            self.logfile.close()
        except Exception as e:
            self.logger_object.log(self.logfile, "Exception while performing BoxCox transformation'. Exception :" + str(e))
            self.logger_object.log(self.logfile, "BoxCox transformation unsuccessful. Exited the method get_trainingData")
            self.logfile.close()
            raise Exception()

        try:
            self.logfile = open(self.logfile_path, mode='a')
            # Dropping columns
            for col in self.drop_cols:
                if col in self.df.columns:
                    self.df = self.df.drop(columns=col, axis=1)
            self.logger_object.log(self.logfile, "Successfully dropped the columns: 'CHAS', 'NOX', 'PTRATIO'")
            self.logger_object.log(self.logfile,"Successfully cleaned the input data")
            self.logfile.close()
        except Exception as e:
            self.logger_object.log(self.logfile, "Exception occured while dropping the columns. Exception :" + str(e))
            self.logger_object.log(self.logfile, "Failed to drop the columns. Exited the method get_trainingData.")
            self.logfile.close()
            raise Exception()

        return self.df
