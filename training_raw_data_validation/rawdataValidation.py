import numpy as np
import pandas as pd
from application_logs.logger import App_Logger
import utils.common_utils as common_utils
from scipy import stats

class RawDataValidation:
    """
        This class shall be used for validate the raw data.\n\n

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """

    def __init__(self, config_path:str):
        self.config = common_utils.read_params(config_path) # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['raw_data_validation'] # this file path help to log the details
        self.logger = App_Logger() # call the App_Logger() to log the details

    def CreateManualRegex(self):
        """
            **Method Name:** CreateManualRegex\n
            **Description:** This method contains a manually defined regex based on the given "FileName" \n
            **Output:** Regex pattern\n
            **On Failure:** Raise Exception\n\n

            :return: Regex pattern
        """
        regex = "['train']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def GetNeumericalFeatures(self, data):
        """
            **Method Name:** GetNeumericalFeatures\n
            **Description:** This method helps to get the neumeric features \n
            **Output:** neumeric features\n
            **On Failure:** Raise Exception\n\n

            :param data: train.csv
            :return: neumeric columns/features
        """
        try:
            self.file = open(self.file_path, 'a+') # open the file
            self.data =  data # read the train.csv file
            self.neumeric_cols = self.data._get_numeric_data().columns # get the neumeric features
            if len(self.neumeric_cols) > 0:
                self.logger.log(self.file, f"Get all Neumeric data type {self.neumeric_cols}")
                self.file.close()
                return self.neumeric_cols  # if present, then return those neumeric columns
            else:
                self.logger.log(self.file, "Neumerical features are not found in dataset")
                self.file.close()
                return self.neumeric_cols # if not present, then return empty list

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}") # logs the error, if error occurs
            self.file.close() # close the file
            raise ex

    def GetCatrgorycalFeatures(self, data):
        """
            **Method Name:** GetCatrgorycalFeatures\n
            **Description:** This method helps to get the categorical features \n
            **Output:** categorical features\n
            **On Failure:** Raise Exception\n\n

            :param data: train.csv
            :return: categorical columns.
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data =  data # read the train.csv file
            self.categorical = self.data.dtypes[
                self.data.dtypes == 'object'].index  # return categorical columns from give dataset.
            if len(self.categorical) > 0:
                self.logger.log(self.file, f"Get all the Categorical data type: {self.categorical}")
                self.file.close()
                return self.categorical  # if present, then return those categorical columns
            else:
                self.logger.log(self.file, "Categorical data are not present in dataset")
                self.file.close()
                return self.categorical # if not present, then return empty list.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def GetLengthofData(self, data):
        """
            **Method Name:** GetLengthofData\n
            **Description:** This method helps to get the length of data \n
            **Output:** length of data\n
            **On Failure:** Raise Exception\n\n

            :param data: train.csv
            :return: length(rows, columns)
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data =  data # read the train.csv file
            self.row_length, self.col_length = self.data.shape[0], self.data.shape[
                1]  # get the row length & column length
            if self.row_length > 0 or self.col_length > 0:
                self.logger.log(self.file,
                                       f"Get the length of data, rows:  {self.row_length}, columns: {self.col_length}")
                self.file.close()
                return self.row_length, self.col_length  # return lenghth of row & col

            else:
                self.logger.log(self.file, "No data is present")
                self.file.close()
                return False, False

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def IsMissingValuePresent(self, data):
        """
            **Method Name:** IsMissingValuePresent\n
            **Description:** This method helps to check is their any missing value present or not. \n
            **Output:** get the missing columns\n
            **On Failure:** Raise Exception\n\n

            :param data: train.csv
            :return: get the missing columns
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data =  data  # read the train.csv file
            self.missing_dataCol = []  # here add only those columns where missing values are present
            self.not_missing_dataCol = []  # here add only those columns where missing values are not present
            for self.col in self.data.columns:
                if self.data[self.col].isnull().sum() > 0:  # check (columns wise, one-by-one) if missing value present
                    self.missing_dataCol.append(self.col)  # append those columns where missing values present.
                else:
                    self.not_missing_dataCol.append(
                        self.col)  # append those columns where missing values are not present

            if len(self.missing_dataCol) > 0:
                self.logger.log(self.file, f"Missing value are present at {self.missing_dataCol}")
                self.file.close()
                return self.missing_dataCol  # return only missing columns.
            else:
                self.logger.log(self.file, "missing values is not present")

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def IsDataImbalanced(self, data, ycol:str):
        """
            **Method Name:** IsDataImbalanced\n
            **Description:** This method helps to check is data balanced or not. \n
            **Output:** True (if balanced), False (if not balanced)\n
            **On Failure:** Raise Exception\n\n

            :param data: train.csv
            :return: True (if balanced), False (if not balanced)
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data  # read the train.csv file
            self.output_col = ycol # mentioned the output column
            self.vals = [] # use fot store the values
            for self.key, self.value in dict(self.data[self.output_col].value_counts()).items():
                self.vals.append(self.value)
            for self.i in range(len(self.vals)):
                if self.vals[self.i] == self.vals[self.i + 1]:  # check the data is balance or not.
                    self.logger.log(self.file, f'Dataset is balanced, {self.data[self.output_col].value_counts()}')
                    return True  # if balance then return True
                    break
                else:
                    self.logger.log(self.file, f'Dataset is not balanced, {self.data[self.output_col].value_counts()}')
                    return False  # if not balance then return False
                    break
            self.file.close()

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def IsOutliersPresent(self, data, cols:list, threshold=3):
        """
            **Method Name:** IsOutliersPresent\n
            **Description:** This method helps to check is outliers present in a particular column.
                             Here I use Z-Score method. \n
            **Output:** outliers_col\n
            **On Failure:** Raise Exception\n\n

            :param data: train.csv
            :param cols: columns
            :param threshold: 3
            :return: outliers_col
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.cols = cols
            self.threshold = threshold
            self.outliers_col = []  # this is used to append the outliers columns.
            for self.col in self.cols:
                self.z = np.abs(stats.zscore(self.data[self.col]))  # to apply the Z-Score for getting the outliers one-by-one columns.
                self.outliers_index = np.where(pd.DataFrame(self.z) > self.threshold)  # get the outliers indexs.
                if len(self.outliers_index[0]) > 0:
                    self.outliers_col.append(self.col)  # appen the columns where outliers are present.
                    self.logger.log(self.file, f"Outliers are present at: {self.col} {self.outliers_index[0]}")
                else:
                    self.logger.log(self.file,
                                           f"Outliers are not present in dataset at: {self.col} {self.outliers_index[0]}")
            self.file.close()
            return self.outliers_col  # return only those columns where outliers are present.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex


if __name__ == '__main__':
    pass

