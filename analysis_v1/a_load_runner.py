import os
from morphomics.io.toml import load_toml, run_toml

dataset = 'v1'

# Path to the toml file. Contains the parameters for each protocol
toml_params_folderpath = "../src/kxa_analysis/toml_params/"
parameters_filename = "load_filter_lm_pi.toml"
parameters_filepath = os.path.join(toml_params_folderpath, parameters_filename)
parameters = load_toml(parameters_filepath)

parameters["Parameters_ID"] = dataset
parameters["Input"]["data_location_filepath"] = f"../data_{dataset}"

mf_name = dataset
parameters["Input"]["morphoframe_name"] = mf_name
parameters["TMD"]["morphoframe_name"] = mf_name
parameters["Filter_frame"]["morphoframe_name"] = mf_name
parameters["Morphometrics"]["morphoframe_name"] = mf_name
parameters["Vectorizations"]["morphoframe_name"] = mf_name

my_pipeline = run_toml(parameters, morphoframe = {})