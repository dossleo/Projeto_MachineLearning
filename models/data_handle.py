import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from msilib.schema import SelfReg
from math import sqrt

class TimeFeatureExtraction():

    def __init__(self,data):
        
        self.data = data
        self.bearing_data = np.array(self.data)
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

class extract_data():
    def __init__(self,
                path = os.path.join(os.getcwd(), "data", "MFPT Fault Data Sets", "1 - Three Baseline Conditions"),
                file = 'baseline_1.mat',
                index_gs = 1):

        self.index_gs = index_gs
        self.path = path
        self.file = file
        self.data = self.mat_to_data(scipy.io.loadmat(os.path.join(self.path, self.file)))

    def mat_to_data(self, mat):
        data = mat["bearing"][0]
        index_gs = data.dtype.names.index('gs')
        data = data[0][index_gs][:,0]
        return data
        
    def ExtractData(self):
        return self.data

    def PlotExample(self):

        self.x = np.linspace(0,1,len(self.data))

        plt.plot(self.x,self.data)
        plt.show()

def get_data(sobre_janela = 0):
    mapped_databases = {
        '1 - Three Baseline Conditions': 'normal',
        '2 - Three Outer Race Fault Conditions': 'outer race',
        '3 - Seven More Outer Race Fault Conditions': 'outer race',
        '4 - Seven Inner Race Fault Conditions': 'inner race'
    }
    features_list = []

    for base in mapped_databases:
        dir_path = os.path.join(os.getcwd(), "data", "MFPT Fault Data Sets", base)

        for file in os.listdir(dir_path):
            if ".mat" in file:
                data = extract_data(path = dir_path, file=file).data
                index = 0

                while index < len(data):
                    splited_data = data[index: index + int(len(data)/16)]
                    time_features = TimeFeatureExtraction(splited_data)
                    features_list.append({
                        'maximum':time_features.maximum(),
                        'minimum':time_features.minimum(),
                        'mean':time_features.mean(),
                        'standard_deviation':time_features.standard_deviation(),
                        'rms':time_features.rms(),
                        'skewness':time_features.skewness(),
                        'kurtosis':time_features.kurtosis(),
                        'form_factor':time_features.form_factor(),
                        'crest_factor':time_features.crest_factor(),
                        'fault': mapped_databases.get(base)
                    })
                    index += int((len(data)/4)*(1-sobre_janela/100))

    return pd.json_normalize(features_list)

