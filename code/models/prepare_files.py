# %%
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class extract_data():
    def __init__(self,
                path = 'C:/Users/leona/Documents/Projeto_MachineLearning/data/MFPT Fault Data Sets/1 - Three Baseline Conditions',
                file = 'baseline_1.mat',
                index_gs = 1):

        self.index_gs = index_gs
        self.path = path
        self.file = file

        self.mat = scipy.io.loadmat(self.path + '/' + self.file)
        self.data = self.mat["bearing"]
        self.data = self.data[0][0][self.index_gs][:,0]

    def ExtractData(self):

        return self.data

    def PlotExample(self):

        self.x = np.linspace(0,1,len(self.data))

        plt.plot(self.x,self.data)
        plt.show()

class label_data():
    def __init__(self,data,label):
        
        self.data = data
        self.label = label

        self.labels = {"normal":0,"outer race":1,"inner race":2,"roller":3}

    def LabelData(self):

        self.d = {"data":self.data,"Defeito":self.label}
        self.df = pd.DataFrame(self.d)

        return self.df

# %%

def run_dataframe():

    paths = ['C:/Users/leona/Documents/Projeto_MachineLearning/data/MFPT Fault Data Sets/1 - Three Baseline Conditions',
            'C:/Users/leona/Documents/Projeto_MachineLearning/data/MFPT Fault Data Sets/2 - Three Outer Race Fault Conditions',
            'C:/Users/leona/Documents/Projeto_MachineLearning/data/MFPT Fault Data Sets/3 - Seven More Outer Race Fault Conditions',
            'C:/Users/leona/Documents/Projeto_MachineLearning/data/MFPT Fault Data Sets/4 - Seven Inner Race Fault Conditions']

    defects = ["normal","outer race","outer race","inner race"]
    coluna_matlab = [1,2,2,2]


    filename = 'OuterRaceFault_1.mat'

    first_df = extract_data(paths[1],filename,coluna_matlab[1])

    data = first_df.ExtractData()

    fisrt_df2 = label_data(data,defects[1])
    df = fisrt_df2.LabelData()


    for i in range(len(paths)):
        for filename in os.listdir(paths[i]):

            dados = extract_data(paths[i],filename,coluna_matlab[i])
            dados = dados.ExtractData()

            label = label_data(dados,defects[i])
            label = label.LabelData()

            df = pd.concat([df,label],ignore_index=True)

    print(df)
    return(df)

run_dataframe()