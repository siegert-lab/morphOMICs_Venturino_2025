from morphomics.io.io import load_obj
from kxa_analysis import plot_2d, merged_dict

def multi_plots_2d(df_path,
                    feature,
                    title, 
                    base_save_folder_path,
                    base_save_filename):


    mf_ = load_obj(df_path)

    # plot everything
    # Replace '+' in the 'Model' column by escaping the '+' character in the filtering step
    mf_['Model'] = mf_['Model'].str.replace(r'\+', '+', regex=True)

    plot_2d(df = mf_, 
            title = title,
            feature = feature, 
            conditions = ['Model', 'Sex'],
            name = base_save_folder_path + base_save_filename + '_kxa-safit2-fkbp5ko_',
            extension = 'html',
            show=False)

    # plot some conditions

    # define names of the files
    name_kxa = base_save_folder_path + base_save_filename + '_kxa_'
    name_kxa_safit_f = base_save_folder_path + base_save_filename + '_kxa-safit_F_'
    name_kxa_safit_m = base_save_folder_path + base_save_filename + '_kxa-safit_M_'

    name_kxa_fkbp5ko_f = base_save_folder_path + base_save_filename + '_kxa-fkbp5ko_F_'
    name_kxa_fkbp5ko_m = base_save_folder_path + base_save_filename + '_kxa-fkbp5ko_M_'

    name_safit = base_save_folder_path + base_save_filename + '_safit_'

    name_fkbp5ko = base_save_folder_path + base_save_filename + '_fkbp5ko_'

    name_list = [name_kxa,
                name_kxa_safit_f,
                name_kxa_safit_m,
                name_kxa_fkbp5ko_f,
                name_kxa_fkbp5ko_m,
                name_safit,
                name_fkbp5ko]

    # define sub morphoframes
    mf_kxa = mf_[mf_['Model'].isin(['1xSaline_4h', '1xKXA_4h'])]

    models_safit = ['1xSaline_4h', '1xKXA_4h', '1xKXA+SAFIT2_4h', '1xSaline+SAFIT2_4h']
    models_fkbp5ko = ['1xSaline_4h', '1xKXA_4h', '1xKXA+FKBP5KO_4h', '1xSaline+FKBP5KO_4h']

    mf_kxa_safit_f = mf_[(mf_['Sex'] == 'F') & (mf_['Model'].isin(models_safit))]
    mf_kxa_safit_m = mf_[(mf_['Sex'] == 'M') & (mf_['Model'].isin(models_safit))]

    mf_kxa_fkbp5ko_f = mf_[(mf_['Sex'] == 'F') & (mf_['Model'].isin(models_fkbp5ko))]
    mf_kxa_fkbp5ko_m = mf_[(mf_['Sex'] == 'M') & (mf_['Model'].isin(models_fkbp5ko))]

    mf_safit = mf_[mf_['Model'].isin(['1xSaline_4h', '1xSaline+SAFIT2_4h'])]
    mf_fkbp5ko = mf_[mf_['Model'].isin(['1xSaline_4h', '1xSaline+FKBP5KO_4h'])]

    mf_list = [
        mf_kxa,
        mf_kxa_safit_f,
        mf_kxa_safit_m,
        mf_kxa_fkbp5ko_f,
        mf_kxa_fkbp5ko_m,
        mf_safit,
        mf_fkbp5ko
    ]

    for sub_mf, name in zip(mf_list, name_list):
        plot_2d(df = sub_mf, 
                title = title,
                feature = feature, 
                conditions=['Model', 'Sex'],
                name = name,
                show=False)


def bt_plots_2d(df_path,
                    feature,
                    title, 
                    base_save_folder_path,
                    base_save_filename):
    mf_ = load_obj(df_path)

    # plot everything
    # Replace '+' in the 'Model' column by escaping the '+' character in the filtering step
    mf_['Model'] = mf_['Model'].str.replace(r'\+', '+', regex=True)

    plot_2d(df = mf_, 
            title = title,
            feature = feature, 
            conditions = ['Model', 'Sex'],
            name = base_save_folder_path + base_save_filename + '_kxa-safit2-fkbp5ko_',
            extension = 'html',
            show=False)
