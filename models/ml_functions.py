import models
import pandas as pd
from .libs.logger import logger
import sklearn.ensemble
from sklearn.model_selection import train_test_split

class MethodPrepare:


    def __init__(self, data:pd.DataFrame, x_columns:list, y_column:str) -> None:
        self.data = data
        self.x_data = self.get_x_data(x_columns)
        self.y_data = self.get_y_data(y_column)

        self.test_size = models.test_size
        self.seed = models.seed

    def get_x_data(self, x_columns:list):
        return self.data[x_columns]

    def get_y_data(self, y_column:list):
        return self.data[y_column]

    def prepare_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_data,
            self.y_data,
            test_size=self.test_size,
            random_state = self.seed
        )

class Classifier(MethodPrepare):

    def __init__(self, data: pd.DataFrame, x_columns: list, y_column: str, classifier:sklearn.ensemble, **kwargs) -> None:
        self.classifier = classifier(**kwargs)
        super().__init__(data, x_columns, y_column)

    @logger
    def run(self):
        self.prepare_data()
        self.classifier.fit(self.x_train,self.y_train)
        self.prediction = self.classifier.predict(self.x_test)
        self.score = self.classifier.score(self.x_test, self.y_test)