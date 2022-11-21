from models import data_vis, data_tools
from rich import pretty, print

pretty.install

# data = data_handle.get_data()

if __name__ == "__main__":
    data = data_tools.DataGenerator().run()
    time_feature_visualization = data_vis.TimeFeatureVisualization(data)
    for i in range(len(data_vis.features)):
        time_feature_visualization.plot_feature(i)