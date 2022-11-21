# General Parameters
mapped_databases = {
        '1 - Three Baseline Conditions': 'normal',
        '2 - Three Outer Race Fault Conditions': 'outer race',
        '3 - Seven More Outer Race Fault Conditions': 'outer race',
        '4 - Seven Inner Race Fault Conditions': 'inner race'
    }

faults = ["normal","outer race","inner race"]

# Data Parameters
frequency_rate_dict = {
    "normal":97656,
    "outer race": 97656,
    "inner race": 48828
}
time_window = 1
overlap = 80

# General Machine Learning Parameters
seed = 30
test_size = 0.25
x_columns = ['maximum', 'minimum', 'mean', 'standard_deviation', 'rms', 'skewness', 'kurtosis', 'form_factor', 'crest_factor']
y_column = 'fault'


# PCA Parameters
