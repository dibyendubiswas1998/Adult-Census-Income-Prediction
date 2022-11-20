from model_training import ModelTraining
from predection import Predection
import utils.common_utils as common_utils
import pandas as pd

if __name__ == '__main__':
    config_path = 'params.yaml'
    mt = ModelTraining(config_path=config_path)
    prd = Predection(config_path=config_path)

    mt.TrainingModel() # train the model
    prd.Predection() # start predection

    config = common_utils.read_params(config_path=config_path)
    data_path = config['predection_data']['predection_data_path']

    data = pd.read_csv(data_path, sep=',')
    print(data.head(10).T)



