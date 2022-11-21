from models import data_vis, data_tools
from rich import pretty, print

pretty.install

# data = data_handle.get_data()

if __name__ == "__main__":
    data = data_tools.DataGenerator().run()
    time_feature_visualization = data_vis.TimeFeatureVisualization(data)
    breakpoint()
    # for metodo in features:
    #     plot = getattr(teste,f"plot_{metodo}")
    #     plot()
