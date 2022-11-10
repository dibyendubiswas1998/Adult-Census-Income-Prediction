import numpy as np
import pandas as pd
from scipy import stats
from imblearn.over_sampling import BorderlineSMOTE
from application_logs.logger import App_Logger
import utils.common_utils as common_utils


class FeaturesEngineering:
    """"
        This class shall be used for handle the raw data.\n\n

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path:str):
        self.config = common_utils.read_params(config_path) # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['testing']['log_files']['features_engineering'] # this file path help to log the details
        self.logger = App_Logger() # call the App_Logger() to log the details

    def ToHandleImbalancedData(self, data, ycol:str):
        """
            **Method Name:** ToHandleImbalancedData\n
            **Description:** This method helps to handle the imbalanced data.
                         Here, we use Borderline-SMOTE to handle the imbalance data.\n
            **Output:** data (after balance).\n
            **On Failure:** Raise Error.\n

        :param data: train.csv
        :param ycol: output_col
        :return: balanced data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data # mention the train.csv file
            self.output_col = ycol # mention the output_col
            self.random_state = self.config['base']['random_state']
            self.bsmote = BorderlineSMOTE(random_state=self.random_state,
                                          kind='borderline-1')  # use BorderLine SMOTE to oversample the data
            self.X = self.data.drop(axis=1, columns=[self.output_col])  # drop the output columns
            self.Y = data[self.output_col]
            self.x, self.y = self.bsmote.fit_resample(self.X, self.Y)
            self.data = self.x
            self.data[self.output_col] = pd.DataFrame(self.y, columns=[self.output_col])
            self.logger.log(self.file, "Handle the imbalanced data using Borderline-SMOTE")
            self.file.close()
            return self.data  # return data (with features & label/output) after oversampling

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def ToHandleOutliers(self, data, col:str, threshold=3):
        """
            **Method Name:** ToHandleOutliers\n
            **Description:** This method helps to handle the outliers using Z-Score.\n
            **Output:** data (after removing the outliers)\n
            **On Failure:** Raise Error.\n

            :param data: train.csv
            :param col: columns
            :param threshold: 3 (default)
            :return: data (without outliers)
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data # mention the train.csv data
            self.col = col # mention the column/columns
            self.threshold = threshold # bydefault we set the threshold value, i.e. 3
            self.z_score = np.abs(stats.zscore(self.data[self.col]))  # to apply the Z-Score to handle the outliers
            self.not_outliers_index = np.where(pd.DataFrame(self.z_score) < self.threshold)[0]  # get the indexes where outliers are not present or ignore outliers indexes
            self.data[self.col] = pd.DataFrame(self.data[self.col]).iloc[
                self.not_outliers_index]  # get the data without outliers
            self.logger.log(self.file, f"Successfully remove the outliers from data {self.col}")
            self.file.close()
            return self.data[self.col]  # return the data without outliers.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def ToRemoveDuplicateValues(self, data):
        """
            **Method Name:** ToRemoveDuplicateValues\n
            **Description:** This method helps to remove the duplicate values\n
            **Output:** data (after removing the duplicate values)\n
            **On Failure:** Raise Error\n

            :param data: train.csv
            :return: data (after removing the duplicate values)
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.logger.log(self.file, f"Before drop the duplicates values, the shape of data is {self.data.shape}")
            self.data = self.data.drop_duplicates()  # simple drop the duplicates values from the given dataset
            self.logger.log(self.file, f"After drop the duplicates values, the shape of data is {self.data.shape}")
            self.logger.log(self.file, "Successfully drop the duplicates values")
            self.file.close()
            return self.data # return the data after removing the duplicate values.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex


    def ToHandleAllMissingValues(self, data, xcols:list):
        """
            **Method Name:** ToHandleAllMissingValues\n
            **Description:** This method helps to handle the missing values. Using this method we replace missing values
                         with mean (of that particular feature).\n
            **Output:** cleaned data\n
            **On Failure:** Raise Error\n

            :param data: data
            :param xcols: xcols
            :return: cleaned data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.xcols = xcols
            if self.xcols is None:
                for self.col in self.data:  # check if column or columns are mention or not.
                    self.data[self.col].dropna(how='all',
                                               inplace=True)  # drop the row if all columns have missing value
                    self.data[self.col].fillna(self.data[self.col].mean(),
                                               inplace=True)  # replace the missing value with mean
                self.logger.log(self.file,
                                       f"Replace the missing value with mean value of {self.data.columns} columns")
                self.file.close()
            else:
                for self.col in self.xcols:
                    self.data[self.col].dropna(how='all',
                                               inplace=True)  # drop the row if all columns have missing value
                    self.data[self.col].fillna(self.data[self.col].mean(),
                                               inplace=True)  # replace the missing value with mean
                self.logger.log(self.file, f"Replace the missing value with mean value of {self.xcols} columns")
                self.file.close()
            return self.data # return the cleaned data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def LabelEncoding(self, data, col:str, mapdct:dict):
        """
            **Method Name:** LabelEncoding\n
            **Description:** This method helps to apply label encoding in output_col\n
            **Output:** encode the output_col\n
            **On Failure:** Raise Error\n

            :param data: train.csv
            :param ycol: ycol
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.col = col
            self.mapdct = mapdct
            self.data[self.col] = self.data[self.col].map(mapdct)
            self.logger.log(self.file, "successfully apply the Label Encoding technique")
            self.file.close()
            return self.data # return data after apply label encoding.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex


    def MeanEncoding(self, data, cols:list, ycol:str):
        """
            **Method Name:** MeanEncoding\n
            **Description:** This method helps to apply mean encoding in the columns\n
            **Output:** data\n
            **On Failure:** Raise Error\n

            :param data: train.csv
            :param cols: columns
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.cols = cols
            self.output_col = ycol
            for self.col in self.cols:
                mean_nominal = self.data.groupby([self.col])[self.output_col].mean().to_dict()
                data[self.col + '_mean_encoding'] = self.data[self.col].map(mean_nominal)
                self.logger.log(self.file, f"Applying the mean encoding on column {self.col}")
                data.drop(axis=1, columns=self.col, inplace=True)
            self.file.close()
            return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def OneHotEncoding(self, data):
        """
            **Method Name:** OneHotEncoding\n
            **Description:** This method helps to apply onehot encoding in the columns\n
            **Output:** data\n
            **On Failure:** Raise Error\n

            :param data: train.csv
            :param cols: columns
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.data = pd.get_dummies(self.data, drop_first=True) # applyt the onehot encosing
            self.logger.log(self.file, "Applying the onehot encoding")
            self.file.close()
            return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def ReplaceZeroValues(self, data, cols:list):
        """
            **Method Name:** ReplaceZeroValues\n
            **Description:** This method helps to replace the zero value with mean value\n
            **Output:** data\n
            **On Failure:** Raise Error\n

            :param data: train.csv
            :param cols: columns
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.cols = cols
            for self.col in self.cols:
                self.data[self.col] = self.data[self.col].replace(0, self.data[self.col].mean()) # replace the zero with mean
                self.logger.log(self.file, f"Replace the value with mean, in {self.col}")
            self.file.close()
            return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex