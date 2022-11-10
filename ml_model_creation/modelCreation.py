from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import utils.common_utils as common_utils
from application_logs.logger import App_Logger


class ModelCreation:
    """
        This class shall be used for create the model

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path: str):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['model_creation']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details


    def ApplyRandomForest(self, x_train, y_train):
        """
            **Method Name:** ApplyRandomForest\n
            **Description:** This method helps to create the randomforest model\n
            **Output:** model\n
            **On Failure:** Raise Error.\n

            :param x_train: x_train data
            :param x_test: y_train data
            :return: model
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.x_train = x_train
            self.y_train = y_train
            # self.grid_params = self.config['ml_algo']['random_forest']['grid_search_cv']
            self.best_params = self.config['ml_algo']['random_forest']['best_params'] # get the best params
            self.n_estimators = self.best_params['n_estimators']
            self.criterion = self.best_params['criterion']
            self.max_depth = self.best_params['max_depth']
            self.max_features = self.best_params['max_features']
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, max_features=self.max_features) # apply RandomForest() algo with best params
            self.clf.fit(self.x_train, self.y_train) # train the model
            self.logger.log(self.file, f"Apply the RandomForest with best params {self.best_params}") # logs the details
            self.file.close()
            return self.clf # return the random forest model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def ApplyXGBoost(self, x_train, y_train):
        """
            **Method Name:** ApplyXGBoost\n
            **Description:** This method helps to apply the XGBoost algo\n
            **Output:** model\n
            **On Failure:** Raise Error.\n

            :param x_train:  x_train data
            :param y_train: y_train data
            :return:
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.x_train = x_train
            self.y_train = y_train
            # self.grid_params = self.config['ml_algo']['xgboost']['grid_search_cv']
            self.best_params = self.config['ml_algo']['xgboost']['best_params']  # get the best params
            self.learning_rate = self.best_params['learning_rate']
            self.max_depth = self.best_params['max_depth']
            self.n_estimators = self.best_params['n_estimators']
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators) # apply the XGBoost classifier algo
            self.xgb.fit(self.x_train, self.y_train) # train the model
            self.logger.log(self.file,
                            f"Apply the RandomForest with best params {self.best_params}")  # logs the details
            self.file.close()
            return self.xgb # return the xgboost model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex