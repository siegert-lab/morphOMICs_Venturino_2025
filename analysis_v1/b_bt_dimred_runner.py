import os
from kxa_analysis import dimreduction_runner, bootstrap_runner
import numpy as np

experiment = 'v1'

# VAE parameters (KL factor)
epochs = 1000
x_values = np.linspace(2, 8, epochs)    
kl_factor_list = list((1 - np.exp(-x_values))/2)
vae_parameters = {}
# Initialize the nested dictionaries
vae_parameters["Dim_reductions"] = {}
vae_parameters["Dim_reductions"]["dimred_method_parameters"] = {}
vae_parameters["Dim_reductions"]["dimred_method_parameters"]["vae"] = {
    "nb_epochs": epochs,
    "kl_factor_function": kl_factor_list
}

# Base path for storing results
output_base_path = f"results/dim_reduction/Morphomics.PID_{experiment}."

# Run combination of pi or lm with umap or vae.
output_filenames = {
    # "pi_pca_vae": "pi_pca_vae_reduced_data.pkl",
    "pi_umap": "pi_umap_reduced_data.pkl",
    "pi_bt_umap": "pi_bt_umap_reduced_data.pkl",
    # "lm_pca_vae": "lm_pca_vae_reduced_data.pkl",
    "lm_umap": "lm_umap_reduced_data.pkl",
    "lm_bt_umap": "lm_bt_umap_reduced_data.pkl"
}

vae_par = {
    'pi_pca_vae' : {
        "nb_epochs": 100,
        "kl_factor_function": list((1 - np.exp(-x_values))/100)
    },

    'lm_pca_vae' :  {
        "nb_epochs": 100,
        "kl_factor_function": list((1 - np.exp(-x_values))/200)
    },
}


pi_lm_filepath = f"results/vectorization/Morphomics.PID_{experiment}.pi_lm.pkl"

# Check and run each method only if the corresponding file does not exist
for key, filename in output_filenames.items():
    file_path = output_base_path + filename
    # if not os.path.exists(file_path):
    if True:
        print(f"Run {key.replace('_', ' ')}")
        is_bt = "bt" in key
        vect_code = key[:2]
        if vect_code == 'lm':
            vect_name = 'morphometrics'
        else:
            vect_name = vect_code
        dimred_name = key[3:]
        if is_bt:
            _ = bootstrap_runner(parameters_id=experiment, 
                    vectors_filepath = pi_lm_filepath,
                     vectorization_name = vect_name,
                        bt_ratio=0.3,
                    )
            input_filepath = f"results/bootstrap/Morphomics.PID_{experiment}.{vect_name}_bt.pkl"
            dimred_name = dimred_name[3:]
        else:
            input_filepath = pi_lm_filepath
            
        if "pca_vae" in key:
            vae_parameters["Dim_reductions"]["dimred_method_parameters"]["vae"] = vae_par[key]

        _ = dimreduction_runner(
            parameters_id=experiment, 
            vectors_filepath=input_filepath, 
            vectorization_name=vect_name, 
            toml_filename=f"{dimred_name}.toml", 
            is_bt=is_bt,
            extra_params=vae_parameters if "pca_vae" in key else {}
        )
    else:
        print(f"Skipping {key}, file already exists: {file_path}")
