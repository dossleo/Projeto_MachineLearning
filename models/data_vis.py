from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from models import faults, frequency_rate_dict, x_columns
from pandas import DataFrame


class raw_data_visualization():
    def __init__(self,raw_data,frequency_rate = frequency_rate_dict["normal"]):
        self.raw_data = raw_data
        frequency_rate = frequency_rate

        self.N = len(self.raw_data)
        self.tempo_total = self.N/frequency_rate

    def tempo_amplitude(self):
        self.vetor_tempo = np.linspace(0,self.tempo_total,self.N)
        plt.plot(self.vetor_tempo,self.raw_data)
        plt.title("Dados Brutos")
        plt.ylabel("Amplitude [gs]")
        plt.show()

class TimeFeatureVisualization():
    def __init__(self,dataframe:DataFrame):
        self.df = dataframe
        self.rows = self.df.shape[0]
        self.columns = self.df.shape[1]
        self.features = x_columns

    def separete_faults(self):
        dataframe = self.df.copy()
        df_normal = dataframe[dataframe["fault"].str.contains("normal")] 
        df_outer = dataframe[dataframe["fault"].str.contains("outer")]      
        df_inner = dataframe[dataframe["fault"].str.contains("inner")]

        self.df_defeitos = [df_normal,df_outer,df_inner]

    def plot_feature(self,index = 0):
        self.separete_faults()

        for defeito in self.df_defeitos:
            plt.plot(range((defeito[self.features[index]].shape[0])),defeito[self.features[index]])
        plt.legend(faults)
        plt.xlabel("Número da Janela Temporal")
        plt.ylabel("Amplitude [gs]")           
        plt.title(self.features[index])
    
        plt.show()
    
    def plot_all(self):
        for i in range(len(self.features)):
            self.plot_feature(i)
