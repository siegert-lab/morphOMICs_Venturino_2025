import os
import re
from morphomics.io.toml import load_toml, run_toml
import collections.abc

def deep_update(original, updates):
    for key, value in updates.items():
        if isinstance(value, collections.abc.Mapping) and key in original:
            deep_update(original[key], value)  # Recursively update nested dict
        else:
            original[key] = value  # Otherwise, overwrite/update key

# Relative path to the toml file. Contains the parameters for each protocol
# This path is going from inside a folder analysis_x/ to the src/kxa_analysis/toml_parameters
default_toml_params_folderpath = "../src/kxa_analysis/toml_params/"

def bootstrap_runner(parameters_id,
                     vectors_filepath,
                     vectorization_name,
                    bt_ratio = 0.60,
                    conditions = ['Layer', 'Model', 'Sex'],
                    toml_params_folderpath = default_toml_params_folderpath, 
                     ):
    
    bt_parameters_filepath = os.path.join(toml_params_folderpath, 'bt.toml')

    bt_parameters = load_toml(bt_parameters_filepath)

    bt_parameters["Parameters_ID"] = parameters_id
    bt_parameters["Bootstrap"]["morphoframe_filepath"] = vectors_filepath
    bt_parameters["Bootstrap"]["feature_to_bootstrap"] = [vectorization_name, "array"]           
    bt_parameters["Bootstrap"]["ratio"] = bt_ratio       
    bt_parameters["Bootstrap"]["bootstrap_conditions"] = conditions

    bt_parameters["Bootstrap"]["bootstrapframe_name"] = parameters_id + '_bt'
    bt_parameters["Bootstrap"]["save_filename"] = vectorization_name + "_bt"

    bt_pipeline = run_toml(bt_parameters, morphoframe={})

    return bt_pipeline


def dimreduction_runner(parameters_id, 
                        vectors_filepath, 
                        vectorization_name, 
                        toml_filename, 
                        is_bt,
                        toml_params_folderpath = default_toml_params_folderpath, 
                        extra_params = {}):
    dimred_parameters_filename = toml_filename
    dimred_parameters_filepath = os.path.join(toml_params_folderpath, dimred_parameters_filename)

    dimred_parameters = load_toml(dimred_parameters_filepath)

    dimred_parameters["Parameters_ID"] = parameters_id
    dimred_parameters["Dim_reductions"]["morphoframe_filepath"] = vectors_filepath

    dimred_parameters["Dim_reductions"]["vectors_to_reduce"] = vectorization_name

    # Extract filename without extension
    dimred_filename_without_extension = os.path.splitext(toml_filename)[0]

    # Create the save_filename
    if is_bt:
        save_filename = f"{vectorization_name}_bt_{dimred_filename_without_extension}"
    else:
        save_filename = f"{vectorization_name}_{dimred_filename_without_extension}"
    dimred_parameters["Dim_reductions"]["save_filename"] = save_filename

    deep_update(dimred_parameters, extra_params)
    dimred_pipeline = run_toml(dimred_parameters, morphoframe={})

    # Return the instance of the class that contains the morphoframe, metadata, and parameters
    return dimred_pipeline




