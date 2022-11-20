import pandas as pd
from rich import pretty, print
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from datetime import datetime
from .libs import logger


class ModelGenerator:

    def __init__(self, data:pd.DataFrame, test_size:float = 0.25, seed:int = 15) -> None:
        self.data = data
        self.test_size = test_size
        self.seed = seed


    def prepare_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data[["x"]],
            self.data["y_prob"],
            test_size=self.test_size,
            random_state = self.seed
        )

    def _internal_logger(self, rede, regressao):
        print(f'[{datetime.today().replace(microsecond=0)}] [red]Realizando testes para o modelo ->[/red] {rede}')
        print(f"[{datetime.today().replace(microsecond=0)}] [green]A acurácia foi de [/green][blue]%.2f%%" % (regressao.score(self.x_test, self.y_test) * 100))


class MethodMLPRegressor(ModelGenerator):

    def _init_(self, data: pd.DataFrame, test_size: float = 0.25, seed: int = 15) -> None:
        super()._init_(data, test_size, seed)

    @logger
    def run(self, rede:tuple):
        self.prepare_data()
        regressao = MLPRegressor(random_state=self.seed, max_iter = 100000, hidden_layer_sizes = rede)
        regressao.fit(self.x_train,self.y_train)
        regressao.score(self.x_test, self.y_test)
        self._internal_logger(rede, regressao)
        self.prediction = regressao.predict(self.x_test)