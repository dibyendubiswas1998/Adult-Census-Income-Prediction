import utils.common_utils as common_utils
from application_logs.logger import App_Logger
from model_training import ModelTraining
from data_preprocessing.preProcessing import PreProcessing
from testing_validation_insertion import Test_Validation
import pandas as pd

class Predection:
    """
        This class shall be used to predect the outcome from model

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['predection']['log_files']['predection_logs']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details
        self.common_utils = common_utils  # load the common utils
        self.model = ModelTraining(config_path=config_path) # get the model name
        self.preprocessing = PreProcessing(config_path=config_path) # load the pre_processing steps
        self.test_val = Test_Validation(config_path=config_path) # apply test validation steps

    def Predection(self):
        """
            **Method Name:** PredectionFromTestData\n
            **Description:** This method helps to predect the outcome from model\n
            **Output:** predected_outcome\n
            **On Failure:** Raise Error.\n

            :param data_path: data path
            :param data: data path
            :return: predected_outcome
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.test_data_path = self.config['artifacts']['split_data']['test_path'] # load the test data path
            self.test_data = pd.read_csv(self.test_data_path, sep=',') # read the data
            self.output_col = self.config['data']['output_col'] # get the output columns
            self.model_path = self.config['model']['model_dir'] # load the model path

            self.model = self.common_utils.load_model(model_path=self.model_path) # get the model
            self.logger.log(self.file, f"Get the best model {self.model}") # logs the details

            self.test_data = self.test_val.Test_Validation() # validate the test data
            self.test_data = self.preprocessing.DropColumn(data=self.test_data, cols=self.output_col) # get the data
            self.logger.log(self.file, "validate the test data for predection")

            self.zero_std_cols_test = self.preprocessing.GetColumnsWithZeroStandardDeviation(data=self.test_data)  # get the zero std dev columns for test data
            self.test_data = self.preprocessing.DropColumn(data=self.test_data, cols=self.zero_std_cols_test)  # drop the zero std dev columns from test data
            self.logger.log(self.file, "drop the zero std columns") # logs the columns

            self.outcome = self.model.predict(self.test_data) # predict the outcome
            self.test_data['outcome'] = self.outcome

            self.predection_data_dir = self.config['predection_data']['predection_data_dir'] # mention the prediction data directory
            self.predection_data_path = self.config['predection_data']['predection_data_path'] # mention the prediction data path
            self.common_utils.clean_prev_dirs_if_exis(dir_path=self.predection_data_dir) # delete or clean the directory
            self.common_utils.create_dir(dirs=[self.predection_data_dir]) # create the predection data directory
            self.common_utils.save_raw_local_data(data=self.test_data, new_data_path=self.predection_data_path) # save the prediction data.csv data
            self.logger.log(self.file, f"save the prediction data in {self.predection_data_dir} directory") # logs the details

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

if __name__ == '__main__':
    prd = Predection('params.yaml')
    prd.Predection()

