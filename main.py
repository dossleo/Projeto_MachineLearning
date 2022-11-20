from models import ml_functions, data_handle, seed
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from models.libs.logger import logger
from rich import pretty, print

pretty.install()

@logger
def main(dataframe:pd.DataFrame = pd.DataFrame()):
    df_data = dataframe

    x_columns = ['maximum', 'minimum', 'mean', 'standard_deviation', 'rms', 'skewness', 'kurtosis', 'form_factor', 'crest_factor']
    y_column = 'fault'
    score = {}

    # Executa a predição
    classifier = ml_functions.Classifier(data = df_data, classifier=RandomForestClassifier, random_state = seed)
    classifier.run()
    name = classifier.classifier.__class__.__name__
    score[f"Classification Score {name}"] = round(classifier.score * 100,2)

    classifier = ml_functions.Classifier(data = df_data, classifier=KNeighborsClassifier)
    classifier.run()
    name = classifier.classifier.__class__.__name__
    score[f"Classification Score {name}"] = round(classifier.score * 100,2)

    classifier = ml_functions.Classifier(data = df_data, classifier=SVC, random_state = seed)
    classifier.run()
    name = classifier.classifier.__class__.__name__
    score[f"Classification Score {name}"] = round(classifier.score * 100,2)

    return score

if __name__ == "__main__":
    data = data_handle.get_data(95)
    print(main(data))