import pandas as pd
import utils.common_utils as common_utils
from application_logs.logger import App_Logger
from testing_data_scaling.dataScaling import DataScaling
from testing_features_engineering.featureEngineering import FeaturesEngineering
from testing_raw_data_validation.rawdataValidation import RawDataValidation


class Test_Validation:
    """
        This class shall be used for validate the data before testing/predection

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path:str):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['testing']['log_files']['testing_main_logs']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details
        self.common_utils = common_utils
        self.raw_data = RawDataValidation(config_path=config_path) # validate the raw data
        self.fea_eng = FeaturesEngineering(config_path=config_path) # apply the feature engineering

    def Test_Validation(self):
        """
            **Method Name:** Test_Validation\n
            **Description:** This method helps to validate the data before testing\n
            **Output:** data\n
            **On Failure:** Raise Error.\n
            
            :return:  test data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            """
                step 1: load the data
            """
            self.test_data_path = self.config['artifacts']['split_data']['test_path'] # load the test data path
            self.test_data = pd.read_csv(self.test_data_path, sep=',') # read the test data
            self.logger.log(self.file, "read the test.csv data") # logs the details

            """
                step 2: validate the test data
            """
            self.neumeric_cols = self.raw_data.GetNeumericalFeatures(data=self.test_data)  # get the neumeric columns
            self.logger.log(self.file, f"Get the neumeric columns {self.neumeric_cols}")  # logs the details

            self.categorical_cols = self.raw_data.GetCatrgorycalFeatures(data=self.test_data)  # get the categorical columns
            self.logger.log(self.file, f"Get the categoricals columns {self.categorical_cols}")  # logs the details

            self.row_length, self.col_length = self.raw_data.GetLengthofData(data=self.test_data)  # get the length of train.csv data
            self.logger.log(self.file, f"Get the length of data {self.row_length, self.col_length}")  # logs the details

            self.missing_value_cols = self.raw_data.IsMissingValuePresent(data=self.test_data)  # get the columns where missing value is present
            self.logger.log(self.file, f"Get the columns where missing value is present {self.missing_value_cols}")  # logs the details

            """
                step 3: apply the features engineering steps
            """
            self.output_col = self.config['data']['output_col']  # read the output column
            self.label_enc = self.config['features_eng']['label_encoding']  # apply label encoding
            self.salary = self.label_enc['salary_col']['col']  # apply label encoding on salary column
            self.mapdct_salary_col = self.label_enc['salary_col']['mapdct_salary_col']  # get the mapping format
            self.sex = self.label_enc['sex_col']['col']  # apply label encoding on sex column
            self.mapdct_sex_col = self.label_enc['sex_col']['mapdct_sex_col']  # get the mapping format

            self.mean_encoding = self.config['features_eng']['mean_encoding']  # apply mean encosing
            self.mean_encoding_cols = self.mean_encoding['mean_encoding_cols']  # apply the mean encoding on particular columns
            self.replace_zero_values_cols = self.config['features_eng']['replace_zero_values_cols']  # get the columns where zero value is present

            self.test_data = self.fea_eng.LabelEncoding(data=self.test_data, col=self.salary, mapdct=self.mapdct_salary_col)  # apply the label encoding on salary column
            self.test_data = self.fea_eng.LabelEncoding(data=self.test_data, col=self.sex, mapdct=self.mapdct_sex_col)  # apply the label encoding on sex column
            self.logger.log(self.file, "Apply the label encoding on salary & sex column")  # logs the details

            self.test_data = self.fea_eng.MeanEncoding(data=self.test_data, cols=self.mean_encoding_cols, ycol=self.output_col)  # apply the mean encoding
            self.logger.log(self.file, f"Apply the mean encoding techniques on {self.mean_encoding_cols} columns")

            self.test_data = self.fea_eng.ReplaceZeroValues(data=self.test_data, cols=self.replace_zero_values_cols)  # replace the zero value with mean value
            self.logger.log(self.file, f"replace the zero values with mean value {self.replace_zero_values_cols} columns")  # logs the details

            return self.test_data  # return test data after validation

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

if __name__ == '__main__':
    tv = Test_Validation('params.yaml')
    tv.Test_Validation()

