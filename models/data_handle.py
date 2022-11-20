from models import mapped_databases
import os
from scipy.io import loadmat

class Read:

    def __init__(self, file_name:str, folder_name:str) -> None:
        self.__check_database(folder_name)
        self.dir_path = os.path.join(os.getcwd(), "data", "MFPT Fault Data Sets", folder_name)
        self.file_path = os.path.join(self.dir_path, file_name)
        self.mat_data =  self.read_mat_data()

    def __check_database__(self, folder_name):
        if not folder_name in mapped_databases:
            raise NotImplementedError("This database system hasn't been implemented yet")

    def mat_data(self):
        return loadmat(self.file_path)

class Prepare:

    def __init__(self, file_name:str, folder_name:str) -> None:
        self.mat_data = Read(file_name, folder_name).mat_data
        self.data = None