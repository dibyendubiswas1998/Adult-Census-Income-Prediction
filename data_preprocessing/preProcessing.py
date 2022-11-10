import pandas as pd
import numpy as np
from application_logs.logger import App_Logger
import utils.common_utils as common_utils
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


class PreProcessing:
    """
        This class shall be used for preprocessed the data.\n\n

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """

    def __init__(self, config_path: str):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['pre_processing']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details

    def DropColumn(self, data, cols:list):
        """
            **Method Name:** DropColumn\n
            **Description:** This method helps to drop the columns\n
            **Output:** data\n
            **On Failure:** Raise Error.\n

            :param data: train.csv or test.csv
            :param cols: columns
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.cols = cols
            if self.cols is None:  # if you can't mention the column(s), then nothing happen
                self.logger.log(self.file, "No columns are droped")
                self.file.close()
                return self.data  # simply return the dta
            else:
                self.data = self.data.drop(axis=1, columns=self.cols)  # drop the column/ columns
                self.logger.log(self.file, f"Successfully drop the columns: {self.cols}")
                self.file.close()
                return self.data  # return data after drop column/ columns

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def SeparateLabelColumn(self, data, ycol:str):
        """
            **Method Name:** SeparateLabelColumn\n
            **Description:** This method helps to separate the label column\n
            **Output:** X, Y\n
            **On Failure:** Raise Error.\n

            :param data: train.csv or test.csv
            :param ycol: ycol
            :return: X, Y
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.output_col = ycol
            self.X = self.data.drop(axis=1, columns=self.output_col)  # separate the features columns
            self.Y = self.data[self.output_col]  # separate the output or label column
            self.logger.log(self.file, f"Successfully drop the column {self.output_col}")
            self.file.close()
            return self.X, self.Y  # return the features & output or label column(s)

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def ImputeMissingValues(self, data):
        """
            **Method Name:** ImputeMissingValues\n
            **Description:** This method replaces all the missing values in the Dataframe using KNN Imputer.\n
            **Output:** data\n
            **On Failure:** Raise Error.\n

            :param data: train.csv or test.csv
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan) # impute the missing value with KNNImputer
            self.new_data = self.imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(self.new_data, columns=self.data.columns)
            self.logger.log(self.file, "Impute the missing values with KNNImputer")
            self.file.close()
            return self.new_data  # return data where no missing values are present

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def GetColumnsWithZeroStandardDeviation(self, data):
        """
            **Method Name:** GetColumnsWithZeroStandardDeviation\n
            **Description:** This method finds out the columns which have a standard deviation of zero.\n
            **Output:** data\n
            **On Failure:** Raise Error.\n

            :param data: train.csv or test.csv
            :return: columns
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.data_describe = self.data.describe()
            self.droping_cols = []
            for self.col in self.data:
                if self.data_describe[self.col]['std'] == 0:  # to check the which column have standard deviation zero
                    self.droping_cols.append(self.col)  # append those columns where std is zero.
            if len(self.droping_cols) > 0:
                self.logger.log(self.file, f"Successfully get the Zero-Standard deviation columns {self.droping_cols}")
                self.file.close()
            else:
                self.logger.log(self.file, f"Not get the Zero-Standard deviation columns {self.droping_cols}")
                self.file.close()
            return self.droping_cols  # return the columns, if you want you can drop those columns.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def TrainTestSplit(self, X, Y):
        """
            **Method Name:** TrainTestSplit\n
            **Description:** This method split the data.\n
            **Output:** x_train, x_test, y_train, y_test\n
            **On Failure:** Raise Error.\n

            :param Xcols: Xcols
            :param Ycol: Ycol
            :return: x_train, x_test, y_train, y_test
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.X = X
            self.Y = Y
            self.test_size = self.config['base']['test_size']
            self.random_state = self.config['base']['random_state']
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)
            self.logger.log(self.file, "Apply the train test split")
            self.file.close()
            return x_train, x_test, y_train, y_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex