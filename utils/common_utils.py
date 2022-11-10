import pickle

import yaml
import os
import shutil



def read_params(config_path: str) -> dict:
    """
        **Method Name:** read_params\n
        **Description:** This method helps to read the parameter from params.yaml file\n
        **Output:** params information\n
        **On Failure:** Raise Exception\n\n

        :param config_path:
        :return: params information from params.yaml file.
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def clean_prev_dirs_if_exis(dir_path: str):
    """
        **Method Name:** clean_prev_dirs_if_exis\n
        **Description:** This method helps to, if any directory is present previously then remove those directories\n
        **On Failure:** Raise Exception\n\n

        :param dir_path: mention the directory path
        :return: remove the directory
    """
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def create_dir(dirs: list):
    """
        **Method Name:** create_dir\n
        **Description:** This method helps to create the directory/directories\n
        **On Failure:** Raise Exception\n\n

        :param dirs: mention the list of directory/directories
        :return: create directory/directories
    """
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def create_file(file_name:str):
    """
        **Method Name:** create_file\n
        **Description:** This method helps to create the file\n
        **On Failure:** Raise Exception\n\n

        :param file_name: mention file name
        :return: create the file
    """
    with open(file_name, 'w') as file:
        pass

def save_raw_local_data(data, new_data_path, header=False):
    """
        **Method Name:** save_raw_local_df\n
        **Description:** This method helps to save the data in local directory\n
        **On Failure:** Raise Exception\n\n

        :param data: data
        :param new_data_path: mention data path, where you can store the data
        :param header: columns header
        :return: save the data
    """
    if header:
        new_col = [col.replace(' ', "_") for col in data.columns]
        data.to_csv(new_data_path, index=False, header=new_col)
    else:
        data.to_csv(new_data_path, index=False)

def save_model(model, model_path:str):
    """
        **Method Name:** save_model\n
        **Description:** This method helps to save the model in a particular local directory\n
        **On Failure:** Raise Exception\n\n

        :param model: moderl name
        :param filename: filename
        :return: model
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    pass
