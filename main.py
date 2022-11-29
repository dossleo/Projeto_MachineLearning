from models import ml_functions, seed
from models.data_tools import DataGenerator
from models.data_vis import TimeFeatureVisualization, RawVisualization, PostProcessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from models.libs.logger import logger
from rich import pretty, print
from sklearn.tree import export_graphviz
import graphviz

pretty.install()

@logger
def main():
    # Criando as entradas
    data_generator = DataGenerator()
    df_data = data_generator.run()
    
    raw_data = data_generator.data
    fault = data_generator.fault
    RawVisualization(raw_data, fault).plt_raw_data()

    # Visualizando os gráficos dos dados de entrada
    time_feature_visualization = TimeFeatureVisualization(df_data)
    time_feature_visualization.plot_all()

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

    PostProcessing.plot_score(score)


    classifier = ml_functions.Classifier(data = df_data, classifier=DecisionTreeClassifier, max_depth=10)
    classifier.run()
    score[f"{classifier.classifier.__class__.__name__}, "] = round(classifier.score * 100,2)
    PostProcessing(classifier, method_name = classifier.classifier.__class__.__name__).plot_confusion_matrix()


    dot_data = export_graphviz(
        classifier.fit_classifier,
        filled = True,
        rounded=True,
        feature_names = classifier.x_columns,
        class_names = ["normal","outer race","inner race"]
    )
    grafico = graphviz.Source(dot_data, filename="data/images/test.gv", format="png")
    grafico.view()
    breakpoint()
    return score

if __name__ == "__main__":
    print(main())