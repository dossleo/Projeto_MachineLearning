from models import ml_functions, data_handle, seed
from models.data_tools import DataGenerator
from models.data_vis import TimeFeatureVisualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from models.libs.logger import logger
from rich import pretty, print

pretty.install()

@logger
def main():
    # Criando as entradas
    df_data = DataGenerator().run()

    time_feature_visualization = TimeFeatureVisualization(df_data)
    time_feature_visualization.plot_all()
    # Criando um dicionario que compara as scores
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
    print(main())