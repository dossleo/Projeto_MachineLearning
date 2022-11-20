from models import ml_functions, data_handle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from models.libs.logger import logger
from models.split_data import get_data

@logger
def main(dataframe:pd.DataFrame = pd.DataFrame()):
    df_data = dataframe

    if len(dataframe) == 0:
        file_name = 'OuterRaceFault_1.mat'
        folder_name = '2 - Three Outer Race Fault Conditions'

        # Lê o arquivo
        mat_data = data_handle.Read(file_name, folder_name).data
        data = mat_data.get("bearing")[0][0][2][:,0]
        df_data = pd.DataFrame({"data":data,"Defeito":"outer race"})

    # Executa a predição
    x_columns = ['maximum', 'minimum', 'mean', 'standard_deviation', 'rms', 'skewness', 'kurtosis', 'form_factor', 'crest_factor']
    y_column = 'fault'
    classifier = ml_functions.Classifier(data = df_data, x_columns=x_columns, y_column=y_column, classifier=RandomForestClassifier)
    classifier.run()
    print("Classification Score: {}%".format(classifier.score * 100))

if __name__ == "__main__":
    data = get_data()
    main(data)
