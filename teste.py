from models import data_handle, data_tools
from rich import pretty, print

pretty.install

# data = data_handle.get_data()

if __name__ == "__main__":
    data = data_tools.DataGenerator().run()
    print(data)