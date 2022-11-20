from code.models import prepare_files, time_features_extraction
import pandas as pd
import os
from models.libs.logger import logger

@logger
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
                data = prepare_files.extract_data(path = dir_path, file=file).data
                index = 0

                while index < len(data):
                    splited_data = data[index: index + int(len(data)/16)]
                    time_features = time_features_extraction.TimeFeatureExtraction(splited_data)
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

if __name__ == "__main__":
    get_data()