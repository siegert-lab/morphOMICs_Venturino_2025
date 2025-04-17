from plot_conditions import multi_plots_2d, bt_plots_2d

df_path = "results/dim_reduction/Morphomics.PID_v1_l.morphometrics_umap_reduced_data"
feature = 'umap'
title = 'UMAP of Morphometrics'
base_folder_path = "results/umap_plot/morphometrics/"
base_save_filename = "lm_umap"

multi_plots_2d(df_path = df_path,
                feature = feature,
                title = title, 
                base_save_folder_path = base_folder_path,
                base_save_filename = base_save_filename)

df_path = "results/dim_reduction/Morphomics.PID_v1_l.morphometrics_pca_vae_reduced_data"
feature = 'pca_vae'
title = 'VAE Latent Space of Morphometrics'
base_folder_path = "results/vae_plot/morphometrics/"
base_save_filename = "lm_vae"

multi_plots_2d(df_path = df_path,
                feature = feature,
                title = title, 
                base_save_folder_path = base_folder_path,
                base_save_filename = base_save_filename)



df_path = "results/dim_reduction/Morphomics.PID_v1_l.morphometrics_bt_umap_reduced_data"
feature = 'umap'
title = 'UMAP of Bootstrapped Morphometrics'
base_folder_path = "results/umap_plot/bt_morphometrics/"
base_save_filename = "lm_bt_umap"

bt_plots_2d(df_path = df_path,
            feature = feature,
            title = title, 
            base_save_folder_path = base_folder_path,
            base_save_filename = base_save_filename)