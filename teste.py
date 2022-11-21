from models import data_handle, data_tools
from rich import pretty, print

pretty.install

# data = data_handle.get_data()

extractor = data_tools.ExtractData()
print(extractor.run())
breakpoint()