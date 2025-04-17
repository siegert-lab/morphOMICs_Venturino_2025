from plot_conditions import multi_plots_2d, bt_plots_2d
experiment = 'v1'

df_path = f"results/dim_reduction/Morphomics.PID_{experiment}.pi_umap_reduced_data"
feature = 'umap'
title = 'UMAP of Persistence Images'
base_folder_path = "results/umap_plot/pi/"
base_save_filename = "pi_umap"

multi_plots_2d(df_path = df_path,
                feature = feature,
                title = title, 
                base_save_folder_path = base_folder_path,
                base_save_filename = base_save_filename)

# df_path = "results/dim_reduction/Morphomics.PID_v1.pi_pca_vae_reduced_data"
# feature = 'pca_vae'
# title = 'VAE Latent Space of Persistence Images'
# base_folder_path = "results/vae_plot/pi/"
# base_save_filename = "pi_vae"

# multi_plots_2d(df_path = df_path,
#                 feature = feature,
#                 title = title, 
#                 base_save_folder_path = base_folder_path,
#                 base_save_filename = base_save_filename)


df_path = f"results/dim_reduction/Morphomics.PID_{experiment}.pi_bt_umap_reduced_data"
feature = 'umap'
title = 'UMAP of Bootstrapped Persistence Images'
base_folder_path = "results/umap_plot/bt_pi/"
base_save_filename = "pi_bt_umap"

bt_plots_2d(df_path = df_path,
            feature = feature,
            title = title, 
            base_save_folder_path = base_folder_path,
            base_save_filename = base_save_filename)