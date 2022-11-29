import numpy as np
from math import sqrt
import os
from scipy.io import loadmat
import models
import pandas as pd

class TimeFeatures():

    def __init__(self,data):
        self.bearing_data = np.array(data)
        self.length = len(self.bearing_data)

    def maximum(self):
        self.max = np.max(self.bearing_data)        
        return self.max

    def minimum(self):
        self.min = np.min(self.bearing_data)        
        return self.min

    def mean(self):
        self.media = np.mean(self.bearing_data)        
        return self.media

    def standard_deviation(self):
        self.std = np.std(self.bearing_data, ddof = 1)        
        return self.std

    def rms(self):
        self.rms_value = sqrt(sum(n*n for n in self.bearing_data)/self.length)        
        return self.rms_value 

    def skewness(self):
        self.n = len(self.bearing_data)
        self.third_moment = np.sum((self.bearing_data - np.mean(self.bearing_data))**3) / self.length
        self.s_3 = np.std(self.bearing_data, ddof = 1) ** 3
        self.skew = self.third_moment/self.s_3

        return self.skew

    def kurtosis(self):
        self.n = len(self.bearing_data)
        self.fourth_moment = np.sum((self.bearing_data - np.mean(self.bearing_data))**4) / self.n
        self.s_4 = np.std(self.bearing_data, ddof = 1) ** 4
        self.kurt = self.fourth_moment / self.s_4 - 3
        return self.kurt

    def crest_factor(self):
        self.cf = self.max/self.rms_value
        return self.cf

    def form_factor(self):
        self.ff = self.rms_value/self.media
        return self.ff


class DataGenerator:

    BASE_DIR = os.path.join(os.getcwd(), "data", "MFPT Fault Data Sets")

    def __init__(self) -> None:
        self.files_list = self.get_file_list()
        self.data = None
        self.fault = None
        self.data_list = []
        self.data_json = []

    def make_file_path(self, file_name:str, folder_name:str):
        return os.path.join(self.BASE_DIR, folder_name, file_name)

    def get_file_list(self) -> list:
        folders = models.mapped_databases.keys()
        files_path_list = []
        for folder in folders:
            files = os.listdir(os.path.join(self.BASE_DIR, folder))
            files_path_list.extend([self.make_file_path(file, folder) for file in files if ".mat" in file])
        return files_path_list

    def read(self, file_path) -> np.array:
        mat_data = loadmat(file_path)
        data = mat_data["bearing"][0]
        index_gs = data.dtype.names.index('gs')
        self.data = data[0][index_gs][:,0]

    def split(self, fault:str):
        index = 0
        self.data_list = []
        data = self.data
        overlap = models.overlap/100
        incrementer = int((models.time_window-models.time_window*overlap)*models.frequency_rate_dict.get(fault))

        while index < len(data):
            self.data_list.append(data[index:index+incrementer])
            index += incrementer
        index -= incrementer
        self.data_list.append(data[index:len(data)])

    def increment_data_json(self, fault: str) -> pd.DataFrame:
        for data in self.data_list:
            time_features = TimeFeatures(data)
            self.data_json.append({
                'maximum':time_features.maximum(),
                'minimum':time_features.minimum(),
                'mean':time_features.mean(),
                'standard_deviation':time_features.standard_deviation(),
                'rms':time_features.rms(),
                'skewness':time_features.skewness(),
                'kurtosis':time_features.kurtosis(),
                'form_factor':time_features.form_factor(),
                'crest_factor':time_features.crest_factor(),
                'fault':fault
            })

    def run(self):
        for file in self.files_list:
            fault = models.mapped_databases.get(file.split("\\")[-2])
            self.fault = fault
            self.read(file)
            self.split(fault)
            self.increment_data_json(fault)
        return pd.json_normalize(self.data_json)
