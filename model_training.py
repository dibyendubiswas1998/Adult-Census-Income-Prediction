import utils.common_utils as common_utils
from application_logs.logger import App_Logger
from training_data_scaling.dataScaling import DataScaling as train_data_scaling
from testing_data_scaling.dataScaling import DataScaling as test_data_scaling
from training_validation_insertion import Train_Validation
from testing_validation_insertion import Test_Validation
from data_preprocessing.preProcessing import PreProcessing
from data_preprocessing.clustering import KMeans_Clustering
from find_best_model.findbestModel import FindBestModel


class ModelTraining:
    """
        This class shall be used for training the model

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['model_training']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details
        self.common_utils = common_utils # load the common utils
        self.train_data_scaling = train_data_scaling(config_path=config_path) # apply the data scaling for train data
        self.test_data_scaling = test_data_scaling(config_path=config_path) # apply the data scaling for test data
        self.train_data_validation = Train_Validation(config_path=config_path) # helps to validate the train data
        self.test_data_validation = Test_Validation(config_path=config_path) # helps to validate the test data
        self.pre_processing = PreProcessing(config_path=config_path) # helps to perform the preprocessing operation
        self.clustering = KMeans_Clustering(config_path=config_path) # helps to create the cluster
        self.find_best_model = FindBestModel(config_path=config_path) # helps to find the best model

    def TrainingModel(self):
        """
            **Method Name:** TrainingModel\n
            **Description:** This method helps to train the model\n
            **Output:** model\n
            **On Failure:** Raise Error.\n

            :param x_train: x_train
            :param y_train: y_train
            :param x_test: x_test
            :param y_test: y_test
            :return: model
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            """
                get the train data after validation
            """
            self.train_data = self.train_data_validation.TrainValidation() # get the train data after validate the data
            self.logger.log(self.file, "Get the train data after validation") # logs the details
            """
                get the test data after validation
            """
            self.test_data = self.test_data_validation.Test_Validation() # get the test data after the validate the data
            self.logger.log(self.file, "Get the test data after validation")  # logs the details
            """
                preprocessed the data both train & test data
            """
            # self.drop_cols = self.config['preProcessing']['drop_cols'] # get the drop columns list
            # self.train_data = self.pre_processing.DropColumn(data=self.train_data, cols=self.drop_cols) # drop the columns from train data
            # self.test_data = self.pre_processing.DropColumn(data=self.test_data, cols=self.drop_cols) # drop the columns from test data
            # self.logger.log(self.file, f"Drop the columns from train & test data") #logs the details

            self.zero_std_cols_train = self.pre_processing.GetColumnsWithZeroStandardDeviation(data=self.train_data)  # get the zero std dev columns for train data
            self.zero_std_cols_test = self.pre_processing.GetColumnsWithZeroStandardDeviation(data=self.test_data)  # get the zero std dev columns for test data
            self.train_data = self.pre_processing.DropColumn(data=self.train_data, cols=self.zero_std_cols_train)  # drop the zero std dev columns from train data
            self.test_data = self.pre_processing.DropColumn(data=self.test_data, cols=self.zero_std_cols_test)  # drop the zero std dev columns from test data
            self.logger.log(self.file, "drop the zero standard deviation columns") # logs the details

            self.output_col = self.config['data']['output_col'] #get the output columns
            self.x_train, self.y_train = self.pre_processing.SeparateLabelColumn(data=self.train_data, ycol=self.output_col) # separate the x_train & y_train data
            self.x_test, self.y_test = self.pre_processing.SeparateLabelColumn(data=self.test_data, ycol=self.output_col) # separate the x_test & y_test data
            self.logger.log(self.file, "separate the x_train, y_train, x_test & y_test data") # logs the details

            self.x_train = self.pre_processing.ImputeMissingValues(data=self.x_train) # impute the missing values using KNN Imputer (for x_train data)
            self.x_test = self.pre_processing.ImputeMissingValues(data=self.x_test) # impute the missing values using KNN Imputer (for x_test data)
            self.logger.log(self.file, "Impute the missing values (x_train & x_test data)") # logs the details
            # """
            #     Apply the KMeans clustering
            # """
            # self.no_cluster = self.clustering.ElbowMethod(data=self.x_train) # get the no of cluster
            # self.logger.log(self.file, f"get the no of cluster {self.no_cluster}") # logs the details
            #
            # self.x_train_cluster_data = self.clustering.CreateCluster(data=self.x_train, no_cluster=self.no_cluster) # create the cluster
            # self.logger.log(self.file, "create the cluster based on x_train data") # logs the details
            #
            # self.cluster_label = self.config['preProcessing']['kmeans_clustering']['cluster_label'] # get the cluster label name
            # self.list_of_clusters  = self.x_train_cluster_data[self.cluster_label].unique() # get the unique cluster
            # self.logger.log(self.file, f"get the unique cluster label, i.e, {self.list_of_clusters}") #logs the details
            #
            # for self.i in self.list_of_clusters:
            #     self.x_train_cluster_data = self.x_train_cluster_data[self.x_train_cluster_data[self.cluster_label] == self.i] # filter the data cluster wise
            #     print(self.x_train_cluster_data)
            #     # self.x_train_cluster_data = self.pre_processing.DropColumn(data=self.x_train_cluster_data, cols=[self.cluster_label]) # drop the cluster label
            #     # print(self.x_train_cluster_data)
            """
                create & get the best model
            """
            # self.x_train = self.train_data_scaling.Standarization(data=self.x_train) # scaled the x_train data
            # self.x_test = self.test_data_scaling.Standarization(data=self.x_test) # scale the x_test data
            # self.logger.log(self.file, "apply the standarization on x_train & x_test data")

            self.model_name, self.model = self.find_best_model.GetBestModel(x_train=self.x_train, y_train=self.y_train, x_test=self.x_test, y_test=self.y_test) # get the best model name & model
            self.logger.log(self.file, f"{self.model_name} is the best model")

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex


if __name__ == '__main__':
    mt = ModelTraining('params.yaml')
    mt.TrainingModel()

