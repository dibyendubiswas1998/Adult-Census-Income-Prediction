import pandas as pd
import utils.common_utils as common_utils
from application_logs.logger import App_Logger
from load_and_split_data.load_split import load_split
from training_data_scaling.dataScaling import DataScaling
from training_raw_data_validation.rawdataValidation import RawDataValidation
from training_features_engineering.featureEngineering import FeaturesEngineering



class Train_Validation:
    """
        This class shall be used for validate the data before training

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path:str):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['training_main_logs']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details
        self.common_utils = common_utils
        self.load_split = load_split(config_path=config_path)
        self.raw_data = RawDataValidation(config_path=config_path)
        self.fea_eng = FeaturesEngineering(config_path=config_path)
        self.data_scaling = DataScaling(config_path=config_path)

    def TrainValidation(self):
        """
            **Method Name:** TrainValidation\n
            **Description:** This method helps to validate the data before training\n
            **Output:** data\n
            **On Failure:** Raise Error.\n

            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            """
                step 1: load & split data
            """
            self.logger.log(self.file, "start the validation before training") # logs the details
            self.load_split.load_and_save_data() # load the data & save the data
            self.load_split.split_data() # split the data to train & test, and save to the folder.
            self.logger.log(self.file, "load & split the data") # logs the details

            """
                step 2: read the data
            """
            self.train_path = self.config['artifacts']['split_data']['train_path'] # get the train.csv data path
            self.train_data = pd.read_csv(self.train_path, sep=',') # read the data
            self.logger.log(self.file, "read the data") # logs the details

            """
                step 3: validate the raw data
            """
            self.neumeric_cols = self.raw_data.GetNeumericalFeatures(data=self.train_data) # get the neumeric columns
            self.logger.log(self.file, f"Get the neumeric columns {self.neumeric_cols}") # logs the details

            self.categorical_cols = self.raw_data.GetCatrgorycalFeatures(data=self.train_data) # get the categorical columns
            self.logger.log(self.file, f"Get the categoricals columns {self.categorical_cols}")  # logs the details

            self.row_length, self.col_length = self.raw_data.GetLengthofData(data=self.train_data) # get the length of train.csv data
            self.logger.log(self.file, f"Get the length of data {self.row_length, self.col_length}")  # logs the details

            self.missing_value_cols = self.raw_data.IsMissingValuePresent(data=self.train_data) # get the columns where missing value is present
            self.logger.log(self.file, f"Get the columns where missing value is present {self.missing_value_cols}") # logs the details

            self.output_col = self.config['data']['output_col'] # read the output column
            self.status = self.raw_data.IsDataImbalanced(data=self.train_data, ycol=self.output_col) # get the status is data is balanced or not
            if self.status: # if True
                self.logger.log(self.file, "Data is balanced") # logs the details
            else: # if False
                self.logger.log(self.file, "Data is not balanced") # logs the details

            """
                step 4: apply the features engineering steps
            """
            self.label_enc = self.config['features_eng']['label_encoding'] # apply label encoding
            self.salary = self.label_enc['salary_col']['col'] # apply label encoding on salary column
            self.mapdct_salary_col = self.label_enc['salary_col']['mapdct_salary_col'] # get the mapping format
            self.sex = self.label_enc['sex_col']['col'] # apply label encoding on sex column
            self.mapdct_sex_col = self.label_enc['sex_col']['mapdct_sex_col'] # get the mapping format

            self.mean_encoding = self.config['features_eng']['mean_encoding'] # apply mean encosing
            self.mean_encoding_cols = self.mean_encoding['mean_encoding_cols'] # apply the mean encoding on particular columns
            self.replace_zero_values_cols = self.config['features_eng']['replace_zero_values_cols'] # get the columns where zero value is present

            self.train_data = self.fea_eng.ToRemoveDuplicateValues(data=self.train_data) # remove the duplicate values
            self.logger.log(self.file, "Remove the duplicate values from data set") # logs the details

            self.train_data = self.fea_eng.LabelEncoding(data=self.train_data, col=self.salary, mapdct=self.mapdct_salary_col) # apply the label encoding on salary column
            self.train_data = self.fea_eng.LabelEncoding(data=self.train_data, col=self.sex, mapdct=self.mapdct_sex_col) # apply the label encoding on sex column
            self.logger.log(self.file, "Apply the label encoding on salary & sex column") # logs the details

            self.train_data = self.fea_eng.MeanEncoding(data=self.train_data, cols=self.mean_encoding_cols, ycol=self.output_col) # apply the mean encoding
            self.logger.log(self.file, f"Apply the mean encoding techniques on {self.mean_encoding_cols} columns")

            # if len(self.missing_value_cols) > 0:
            #     self.data = self.fea_eng.ToHandleAllMissingValues(data=self.data,
            #                                                       xcols=self.missing_value_cols)  # handle the missing values
            #     self.logger.log(self.file, "Handle the missing values")  # logs the details

            self.train_data = self.fea_eng.ReplaceZeroValues(data=self.train_data, cols=self.replace_zero_values_cols) # replace the zero value with mean value
            self.logger.log(self.file, f"replace the zero values with mean value {self.replace_zero_values_cols} columns") # logs the details

            self.train_data = self.fea_eng.ToHandleImbalancedData(data=self.train_data, ycol=self.output_col)  # balanced the data
            self.logger.log(self.file, "Balanced the data")  # logs the details
            return self.train_data # return data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

