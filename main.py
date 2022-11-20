from models import ml_functions, data_handle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from models.libs.logger import logger

@logger
def main():
    file_name = 'OuterRaceFault_1.mat'
    folder_name = '2 - Three Outer Race Fault Conditions'
    x_columns = ["data"]
    y_column = 'Defeito'

    # Lê o arquivo
    mat_data = data_handle.Read(file_name, folder_name).data
    data = mat_data.get("bearing")[0][0][2][:,0]
    df_data = pd.DataFrame({"data":data,"Defeito":"outer race"})

    # Executa a predição
    classifier = ml_functions.Classifier(data = df_data, x_columns=x_columns, y_column=y_column, classifier=RandomForestClassifier)
    classifier.run()
    print("Classification Score: {}%".format(classifier.score * 100))

if __name__ == "__main__":
    main()
