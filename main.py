from load_and_split_data.load_split import load_split
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        # stages: load_and_split_data
        lo = load_split(config_path=parsed_args.config)
        lo.load_and_save_data()
        lo.split_data()


        # stages: training_raw_data_validation


    except Exception as ex:
        raise ex




