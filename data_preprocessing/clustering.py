import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from application_logs.logger import App_Logger
import utils.common_utils as common_utils


class KMeans_Clustering:
    """
        This class shall  be used to divide the data into clusters before training/tesing.

        Written By: Dibyendu Biswas.\n\n
        Version: 0.0.1\n\n
    """
    def __init__(self, config_path: str):
        self.config = common_utils.read_params(config_path)  # read the information from params.yaml file as dict form
        self.file_path = self.config['execution_logs']['training']['log_files']['pre_processing']  # this file path help to log the details
        self.logger = App_Logger()  # call the App_Logger() to log the details

    def ElbowMethod(self, data):
        """
            **Method Name:** ElbowMethod\n
            **Description:** This method helps to saves the plot to decide the optimum number of clusters to the file.\n
            **Output:** data\n
            **On Failure:** Raise Error.\n

            :param data: train.csv or test.csv
            :return: data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.wcss = []
            for i in range(1, 11):
                self.kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                self.kmeans.fit(self.data)
                self.wcss.append(self.kmeans.inertia_)
            plt.plot(range(1, 11), self.wcss)  # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            # plt.show()
            self.preprocessing_dir = self.config['preProcessing']['preprocessing_dir'] # mention the preprocessed directory
            self.kmeans_elbow_dir = self.config['preProcessing']['kmeans_clustering']['kmeans_elbow_dir'] # mention kmeans_elbow directory
            common_utils.clean_prev_dirs_if_exis(self.preprocessing_dir) # delete the directory if it's exists
            common_utils.create_dir([self.preprocessing_dir, self.kmeans_elbow_dir]) # create the directory
            self.kmeans_elbow_img_path = self.config['preProcessing']['kmeans_clustering']['kmeans_elbow_img_path'] # mention the kmeans_elbow_img path
            plt.savefig(self.kmeans_elbow_img_path) # to save the graph (wcs vs no. of cluster)
            self.logger.log(self.file, f"Save the KMeans_Elbow graph in {self.kmeans_elbow_img_path} directory")
            self.kn = KneeLocator(range(1, 11), self.wcss, curve='convex', direction='decreasing')  # KneeLocator helps to get the number of cluster
            self.logger.log(self.file, f"Get the Number of clusters using KMeans Clustering, i.e. {self.kn.knee}")
            self.file.close()
            return self.kn.knee  # get or return the number of cluster

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex

    def CreateCluster(self, data, no_cluster):
        """
            **Method Name:** CreateCluster\n
            **Description:** This method helpa to create the clusters\n
            **Output:** data\n
            **On Failure:** Raise Error.\n

            :param data: train.csv or test.csv
            :param no_cluster: no of cluster
            :return: cluster data
        """
        try:
            self.file = open(self.file_path, 'a+')  # open the file
            self.data = data
            self.no_cluster = no_cluster
            self.kmeans = KMeans(n_clusters=self.no_cluster, init='k-means++', random_state=101)  # create a cluster using KMeans Clustering
            self.model_dir = self.config['model']['model_dir'] # mention the model directory
            self.Kmeans_dir = self.config['model']['Kmeans_dir'] # mention the KMeansdirectoryt
            self.Kmeans_path = self.config['model']['Kmeans_path'] # mention the model path
            common_utils.create_dir([self.model_dir, self.Kmeans_dir]) # create the model & KMeans directory
            common_utils.save_model(model=self.kmeans, model_path=self.Kmeans_path) # save the model
            self.y_kmeans = self.kmeans.fit_predict(self.data)  # predict the cluster labels
            self.cluster_label = self.config['preProcessing']['kmeans_clustering']['cluster_label']  # get the cluster label name
            self.data[self.cluster_label] = self.y_kmeans  # attach the cluster labels with the given data
            self.cluster_data_dir = self.config['preProcessing']['kmeans_clustering']['cluster_data_dir'] # mention the cluster_dasta directory
            self.cluster_data_path = self.config['preProcessing']['kmeans_clustering']['cluster_data_path'] # mention cluster_data_path
            common_utils.create_dir([self.cluster_data_dir]) # create the cluster data directory
            common_utils.save_raw_local_data(data=self.data, new_data_path=self.cluster_data_path) # save the cluster data in a preprocessed directory
            self.logger.log(self.file, "Successfully create the clusters & labeled the cluster")
            self.file.close()
            return self.data # return the cluster labeled data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')  # open the file
            self.logger.log(self.file, f"Error: {ex}")  # logs the error, if error occurs
            self.file.close()  # close the file
            raise ex