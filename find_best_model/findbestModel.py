import utils.common_utils as common_utils
from application_logs.logger import App_Logger
from ml_model_creation.modelCreation import ModelCreation
from sklearn.metrics import roc_auc_score, accuracy_score


class FindBestModel:
    """
        This class shall be used for find the best model

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path: str):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['find_best_model']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details
        self.model = ModelCreation(config_path=config_path) # get the model

    def GetBestModel(self, x_train, y_train, x_test, y_test):
        """
            **Method Name:** GetBestModel\n
            **Description:** This method helps to get the best model\n
            **Output:** best model\n
            **On Failure:** Raise Error.\n

            :param x_train: x_train
            :param y_train: y_train
            :param x_test: x_test
            :param y_test: y_test
            :return: best model
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
            # For XGBoost Classifier:
            self.xgb = self.model.ApplyXGBoost(x_train=self.x_train, y_train=self.y_train) # get the xgboost model
            self.prediction_xgboost = self.xgb.predict(self.x_test)  # Predictions using the XGBoost Model
            if len(self.y_test.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(self.y_test, self.prediction_xgboost)
                self.logger.log(self.file, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(self.y_test, self.prediction_xgboost)  # AUC for XGBoost
                self.logger.log(self.file, 'AUC for XGBoost:' + str(self.xgboost_score))  # Log AUC

            # For RandomForest Classifier:
            self.ran = self.model.ApplyRandomForest(x_train=self.x_train, y_train=self.y_train) # get the random forest model
            self.prediction_random_forest = self.ran.predict(self.x_test)  # prediction using the Random Forest Algorithm
            if len(self.y_test.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(self.y_test, self.prediction_random_forest)
                self.logger.log(self.file, 'Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(self.y_test, self.prediction_random_forest)  # AUC for Random Forest
                self.logger.log(self.file, 'AUC for RF:' + str(self.random_forest_score))

            # Comparing the two models:
            if (self.random_forest_score < self.xgboost_score):
                self.xgboost_dir = self.config['model']['xgboost_dir'] # mention the directory
                self.xgboost_path = self.config['model']['xgboost_path'] # mention the model path
                common_utils.create_dir(dirs=[self.xgboost_dir]) # create the xgboost directory
                common_utils.save_model(model=self.xgb, model_path=self.xgboost_path)  # save the model
                self.logger.log(self.file, "XGBoost is the best model & save the model in model directory")
                self.file.close()
                return 'XGBoost', self.xgb
            else:
                self.random_forest_dir = self.config['model']['random_forest_dir']  # mention the directory
                self.random_forest_path = self.config['model']['random_forest_path']  # mention the model path
                common_utils.create_dir(dirs=[self.random_forest_dir])  # create the xgboost directory
                common_utils.save_model(model=self.ran, model_path=self.random_forest_path)  # save the model
                self.logger.log(self.file, "RandomForest is the best model & save the model in model directory")
                self.file.close()
                return 'RandomForest', self.ran

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex