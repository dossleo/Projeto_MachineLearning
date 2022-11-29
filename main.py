from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

from models import ml_functions, seed
from models.data_tools import DataGenerator
from models.data_vis import (PostProcessing, RawVisualization,
                             TimeFeatureVisualization)
from models.libs.logger import logger


@logger
def main():
    # Criando as entradas
    data_generator = DataGenerator()
    df_data = data_generator.run()
    
    raw_data = data_generator.data
    fault = data_generator.fault
    RawVisualization(raw_data, fault).plt_raw_data()

    # Visualizando os gráficos dos dados de entrada
    # time_feature_visualization = TimeFeatureVisualization(df_data)
    # time_feature_visualization.plot_all()

    # Criando um dicionario que compara as scores
    score = {}
    RandomForestClassifier()

    # Executa a predição
    classifier = ml_functions.Classifier(data = df_data, classifier=RandomForestClassifier, random_state = seed)
    classifier.run()
    score[f"{classifier.classifier.__class__.__name__}"] = round(classifier.score * 100,2)
    PostProcessing(classifier, method_name = classifier.classifier.__class__.__name__).plot_confusion_matrix()

    classifier = ml_functions.Classifier(data = df_data, classifier=KNeighborsClassifier)
    classifier.run()
    score[f"{classifier.classifier.__class__.__name__}"] = round(classifier.score * 100,2)
    PostProcessing(classifier, method_name = classifier.classifier.__class__.__name__).plot_confusion_matrix()

    classifier = ml_functions.Classifier(data = df_data, classifier=SVC, random_state = seed)
    classifier.run()
    score[f"{classifier.classifier.__class__.__name__}"] = round(classifier.score * 100,2)
    PostProcessing(classifier, method_name = classifier.classifier.__class__.__name__).plot_confusion_matrix()

    classifier = ml_functions.Classifier(data = df_data, classifier=GaussianNB)
    classifier.run()
    score[f"{classifier.classifier.__class__.__name__}"] = round(classifier.score * 100,2)
    PostProcessing(classifier, method_name = classifier.classifier.__class__.__name__).plot_confusion_matrix()

    classifier = ml_functions.Classifier(data = df_data, classifier=NuSVC, random_state = seed)
    classifier.run()
    score[f"{classifier.classifier.__class__.__name__}"] = round(classifier.score * 100,2)
    PostProcessing(classifier, method_name = classifier.classifier.__class__.__name__).plot_confusion_matrix()

    classifier = ml_functions.Classifier(data = df_data, classifier=DecisionTreeClassifier, criterion = "entropy")
    classifier.run()
    DecisionTreeClassifier()
    score[f"{classifier.classifier.__class__.__name__}, "] = round(classifier.score * 100,2)
    post_processing = PostProcessing(classifier, method_name = classifier.classifier.__class__.__name__)
    post_processing.plot_confusion_matrix()

    try:
        post_processing.plot_decision_tree()
    except:
        print("Error: Grahpviz not found")
        print("The system not found graphviz software. Please install graphviz from http://www.graphviz.org/download/")

    PostProcessing.plot_score(score)

    return score

if __name__ == "__main__":
    main()
