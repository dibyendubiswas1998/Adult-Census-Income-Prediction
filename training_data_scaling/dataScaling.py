from sklearn.preprocessing import StandardScaler
from application_logs.logger import App_Logger
import utils.common_utils as common_utils


class DataScaling:
    """
        This class shall be used for hscaling the data.\n\n

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path:str):
        self.config = common_utils.read_params(config_path) # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['data_scaling'] # this file path help to log the details
        self.logger = App_Logger() # call the App_Logger() to log the details

    def Standarization(self, data):
        """
            **Method Name:** Standarization\n
            **Description:** This method helps to standarized the data\n
            **Output:** data\n
            **On Failure:** Raise Error.\n

            :param data: train.csv
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.stadarize = StandardScaler() # apply the standarization
            self.scaled_data = self.stadarize.fit_transform(self.data)
            self.logger.log(self.file, f"Standarized the data using StandardScaler() technique")
            self.file.close()
            return self.scaled_data # return the scaled data, where std:1, mean:0

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex