from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from models import faults, frequency_rate_dict, x_columns
from pandas import DataFrame
from sklearn.metrics import ConfusionMatrixDisplay
import os

def create_images_dir():
    dir_path = os.path.join('data/images')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path

BASE_PATH = create_images_dir()


class RawVisualization():
    def __init__(self,raw_data,fault):

        self.raw_data = raw_data
        self.fault = fault
        frequency_rate = frequency_rate_dict.get(fault)
        self.N = len(self.raw_data)
        self.tempo_total = self.N/frequency_rate

    def plt_raw_data(self):
        self.vetor_tempo = np.linspace(0,self.tempo_total,self.N)
        plt.plot(self.vetor_tempo,self.raw_data)
        plt.title(f"Dados Brutos - {self.fault}")
        plt.ylabel("Amplitude [gs]")
        plt.savefig(F"{BASE_PATH}/raw_data.png")
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
        plt.savefig(F"{BASE_PATH}/{self.features[index]}.png")
        plt.show()
    
    def plot_all(self):
        for i in range(len(self.features)):
            self.plot_feature(i)

class PostProcessing():

    def __init__(self, classifier, method_name) -> None:
        self.classifier = classifier
        self.title = method_name
        pass

    def plot_confusion_matrix(self):
        disp = ConfusionMatrixDisplay.from_estimator(
            self.classifier.fit_classifier,
            self.classifier.x_test,
            self.classifier.y_test,
            display_labels=faults,
            cmap=plt.cm.Blues,
            normalize='true',
        )
        disp.ax_.set_title(f"Matriz de Confusão - {self.title}")
        plt.savefig(F"{BASE_PATH}/{self.title}.png")

        plt.show()