import numpy as np
import pandas as pd
from application_logs.logger import App_Logger
import utils.common_utils as common_utils
from sklearn.model_selection import train_test_split


class load_split:
    """
        This class is used for load the data & split the data\n\n

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """

    def __init__(self, config_path:str):
        self.config = common_utils.read_params(config_path) # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['file_operation'] # this file path help to log the details
        self.logger = App_Logger() # call the App_Logger() to log the details


    def load_and_save_data(self):
        """
            **Method Name:** load_and_save_data\n
            **Description:** This method helps to load the data from particular location & save the data in a local directory\n
            **Output:** save data\n
            **On Failure:** Raise Exception\n\n

            :param config_path: params.yaml file
            :return: save data
        """
        try:
            self.file = open(self.file_path, 'a+') # open the file
            self.github_url = self.config['data_source']['github_url'] # read/load the github url
            self.data = pd.read_csv(self.github_url, sep=',') # read the data

            self.artifacts = self.config['artifacts']
            self.artifacts_dir = self.artifacts['artifacts_dir'] # mention the artifacts directory
            self.raw_local_data_dir = self.artifacts['raw_local_data_dir'] # mention the raw_local_data_dir directory
            self.raw_local_data_path = self.artifacts['raw_local_data'] # mention the raw_local_data path

            common_utils.clean_prev_dirs_if_exis(self.artifacts_dir) # delete the directory if it is previously created.
            self.logger.log(self.file, f"delete the {self.artifacts_dir} directory if it is previously created")
            common_utils.create_dir([self.artifacts_dir, self.raw_local_data_dir]) # create the artifacts and raw_local_data directories
            self.logger.log(self.file, f"create the {self.artifacts_dir, self.raw_local_data_dir} directories to store the data in local")
            common_utils.save_raw_local_data(self.data, self.raw_local_data_path) # save data data to raw_local_data directory folder
            self.logger.log(self.file, f"save the data in {self.raw_local_data_path} ")
            self.file.close() # close the file

        except Exception as e:
            raise e


    def split_data(self):
        """
            **Method Name:** split_data\n
            **Description:** This method helps to split the data based on split_ratio\n
            **Output:** save data in train.csv & test.csv\n
            **On Failure:** Raise Exception\n\n

            :param config_path: params.yaml file
            :return: split data in train.csv & test.csv
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.base = self.config['base']  # mention the base
            self.split_ratio = self.base['test_size']  # mention the split_ratio
            self.random_state = self.base['random_state']  # mention the random_state

            self.artifacts = self.config['artifacts']  # mention the artifacts path
            self.raw_local_data_path = self.artifacts['raw_local_data']  # mention the raw_local_data data path to read the data
            self.split_data = self.artifacts['split_data']  # mention the split_data path
            self.processed_data_dir = self.split_data[
                'processed_data_dir']  # mention the processed_data_dir directory path to save the train.csv & test.csv data
            self.train_data_path = self.split_data['train_path']  # mention the train_data path
            self.test_data_path = self.split_data['test_path']  # mention the test_data path

            common_utils.create_dir([self.processed_data_dir])  # create the processed_data_dir directory
            self.logger.log(self.file, f"create the directory {self.processed_data_dir} to store the train & test data")
            self.data = pd.read_csv(self.raw_local_data_path, sep=',')  # read the data from local directory path
            self.logger.log(self.file, f"read the data from {self.raw_local_data_path}")
            self.train, self.test = train_test_split(self.data, test_size=self.split_ratio,
                                           random_state=self.random_state)  # split the data based on spliting ratio
            self.logger.log(self.file, f"split the train & test data based on split_ratio:{self.split_ratio}, random_state:{self.random_state}")
            for self.data, self.data_path in (self.train, self.train_data_path), (self.test, self.test_data_path):  # it helps to save the train & test data in aparticular directory
                common_utils.save_raw_local_data(self.data, self.data_path)
            self.logger.log(self.file, f"store the data in {self.train_data_path, self.test_data_path}")

        except Exception as e:
            raise e


if __name__ == "__main__":
    pass

