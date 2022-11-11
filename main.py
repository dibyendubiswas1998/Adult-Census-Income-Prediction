from load_and_split_data.load_split import load_split
from training_raw_data_validation.rawdataValidation import RawDataValidation
import utils.common_utils as common_utils
import pandas as pd
import argparse

if __name__ == "__main__":
    config = common_utils.read_params('params.yaml')
    train_path = config['artifacts']['split_data']['train_path']
    data = pd.read_csv(train_path, sep=',')

    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        # stages: load_and_split_data
        lo = load_split(config_path=parsed_args.config)
        lo.load_and_save_data()
        lo.split_data()


        # stages: training_raw_data_validation
        rw = RawDataValidation(config_path=parsed_args.config)


    except Exception as ex:
        raise ex




