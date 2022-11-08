# %%
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class extract_data():
    def __init__(self,
                path = 'C:/Users/leona/Documents/Projeto_MachineLearning/data/MFPT Fault Data Sets/1 - Three Baseline Conditions',
                file = '/baseline_1.mat'):

        self.path = path
        self.file = file

        self.mat = scipy.io.loadmat(self.path + self.file)
        self.data = self.mat["bearing"]
        self.data = self.data[0][0][1][:,0]

    def ExtractData(self):

        return self.data

    def PlotExample(self):

        self.x = np.linspace(0,1,len(self.data))

        plt.plot(self.x,self.data)
        plt.show()

class label_data():
    def __init__(self,data,label = "normal"):
        self.data = data
        self.label = label

    def LabelData(self):

        self.d = {"data":self.data,"Defeito":self.label}
        self.df = pd.DataFrame(self.d)

        return self.df

# %%
test = extract_data()

data = test.ExtractData()

test2 = label_data(data)
print(test2.LabelData())
