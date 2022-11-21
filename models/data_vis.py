from matplotlib import pyplot as plt
import numpy as np
<<<<<<< Updated upstream
from . import data_handle
=======
>>>>>>> Stashed changes
import seaborn as sns
from models import faults, frequency_rate_dict
from pandas import DataFrame

features = ["maximum","minimum","mean","standard_deviation","rms","skewness","kurtosis","form_factor","crest_factor"]

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

    def separete_faults(self):
        dataframe = self.df.copy()
        df_normal = dataframe[dataframe["fault"].str.contains("normal")] 
        df_outer = dataframe[dataframe["fault"].str.contains("outer")]      
        df_inner = dataframe[dataframe["fault"].str.contains("inner")]

        self.df_defeitos = [df_normal,df_outer,df_inner]

    def plot_feature(self,index = 0):
        self.separete_faults()

        for defeito in self.df_defeitos:
            plt.plot(range((defeito[features[index]].shape[0])),defeito[features[index]])
        plt.legend(faults)
        plt.xlabel("Número da Janela Temporal")
        plt.ylabel("Amplitude [gs]")           
        plt.title(features[index])
    
        plt.show()


<<<<<<< Updated upstream
if __name__ == "__main__":

    df = data_handle.get_data(0)

    teste = time_feature_visualization(df)
    for i in range(len(features)):
        teste.plot_feature(i)
    print(df)
=======
# if __name__ == "__main__":
#     df = data_handle.get_data(0)

#     teste = TimeFeatureVisualization(df)
#     for metodo in features:
#         plot = getattr(teste,f"plot_{metodo}")
#         plot()

#     breakpoint()
#     print(df)
>>>>>>> Stashed changes
