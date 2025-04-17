from morphomics.io.io import load_obj
from kxa_analysis import plot_2d, merged_dict

experiment = 'human_tl'

df_path = f"results/dim_reduction/Morphomics.PID_{experiment}.morphometrics_umap_reduced_data"
feature = 'umap'
title = 'UMAP of Morphometrics'
base_folder_path = "results/umap_plot/morphometrics/"
base_save_filename = "lm_umap"

mf_ = load_obj(df_path)

name_kxa = base_folder_path + base_save_filename + '_kxa_'


plot_2d(df = mf_, 
        title = title,
        feature = feature, 
        conditions = ['Model', 'Sex'],
        name = name_kxa,
        extension = 'html',
        show=False)

plot_2d(df = mf_, 
        title = title,
        feature = feature, 
        conditions = ['Model', 'Sex'],
        name = name_kxa,
        extension = 'pdf',
        show=False)


df_path = f"results/dim_reduction/Morphomics.PID_{experiment}.morphometrics_bt_umap_reduced_data"
feature = 'umap'
title = 'UMAP of Bootstrapped Morphometrics'
base_folder_path = "results/umap_plot/bt_morphometrics/"
base_save_filename = "lm_bt_umap"

name_bt_kxa = base_folder_path + base_save_filename + '_kxa_'
mf_ = load_obj(df_path)

plot_2d(df = mf_, 
        title = title,
        feature = feature, 
        conditions = ['Model', 'Sex'],
        name = name_bt_kxa,
        extension = 'html',
        show=False)

plot_2d(df = mf_, 
        title = title,
        feature = feature, 
        conditions = ['Model', 'Sex'],
        name = name_bt_kxa,
        extension = 'pdf',
        show=False)