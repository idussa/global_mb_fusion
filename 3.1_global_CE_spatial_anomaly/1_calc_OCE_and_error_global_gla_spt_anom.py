"""
Calculate the observational consensus estimate for every individual glacier

calc_OCE_and_error_global_gla_reg_anom.py

Author: idussa
Date: Feb 2021
Last changes: Feb 2021

Scripted for Python 3.7

Description:
This script reads glacier-wide mass balance data edited from WGMS FoG database
and regional glacier anomalies produced by calc_regional_anomalies_and_error.py
and provides the observational consensus estimate for every individual glacier
with available geodetic observations WGMS Id

Input:  GEO_MASS_BALANCE_DATA_20200824.csv
        Regional_anomalies_ref_period_2009-2018.csv
        (UTF-8 encoding)

Return: tbd.svg
"""


import os
import math
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import fnmatch
from functions_ggmc import *
pd.options.mode.chained_assignment = None  # default='warn'
import scipy  # usage: scipy.stats.funct()

fin_hydro_yr= {'ALA' : 0.75,'WNA' : 0.75,'ACN' : 0.75,'ACS' : 0.75,'GRL' : 0.75,
           'ISL' : 0.75,'SJM' : 0.75,'SCA' : 0.75,'RUA' : 0.75,'ASN' : 0.75,
           'CEU' : 0.75,'CAU' : 0.75,'ASC' : 0.75,'ASW' : 0.75,'ASE' : 0.75,
           'TRP' : 0,'SA1' : 0.25,'SA2' : 0.25,'NZL' : 0.25,'ANT' : 0.25}

ini_hydro_yr= {'ALA' : 0.25,'WNA' : 0.25,'ACN' : 0.75,'ACS' : 0.75,'GRL' : 0.75,
           'ISL' : 0.25,'SJM' : 0.25,'SCA' : 0.25,'RUA' : 0.25,'ASN' : 0.25,
           'CEU' : 0.25,'CAU' : 0.25,'ASC' : 0.25,'ASW' : 0.25,'ASE' : 0.25,
           'TRP' : 0,'SA1' : 0.75,'SA2' : 0.75,'NZL' : 0.75,'ANT' : 0.75}


##########################################
##########################################
"""main code"""
##########################################
##########################################

# Define input
##### 1. PATH TO FILES ######

path = os.path.dirname(os.path.abspath(__file__))
# path_proj = os.path.join(path, "..")
path_proj = 'C:\\Users\\idussail2\\Documents\\PROJECTS\\G3P_project\\codes\\mb_data_crunching_local\\'

fog_version = '2024-01'
yr_ini= 1915 # define begining year of anomaly files, determined by longer anomally from CEU starting in 1915
yr_end= 2023 # define the end year with data, determied by the last call for data in WGMS

# read file with global geodetic mass-balance data and regional anomalies
in_data_geo = os.path.join(path, 'in_data', 'fog-'+fog_version,'_FOG_GEO_MASS_BALANCE_DATA_'+fog_version+'.csv')
path_spt_anom = os.path.join(path_proj, '2.1_spatial_anomalies', 'out_data_'+fog_version+'_review', 'LONG-NORM_spatial_gla_anom_ref_2011-2020')# path to spatial anomaly files
# in_data_anom_err = path_proj + '2.1_spatial_anomalies\\out_data\\UNC_spatial_gla_anom_ref_2011-2020\\'# path to spatial anomaly uncertainty files


out_dir = os.path.join(path, 'out_data_'+fog_version+'_review', 'OCE_files_by_region')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

##### 2. DEFINE GEODETIC DATA ######

# # read global geodetic mass-balance data from csv into dataframe
geo_df= pd.read_csv(in_data_geo, encoding='latin1', delimiter=',', header=0, index_col='WGMS_ID').sort_index()
geo_df.reset_index(inplace=True)

##### 3. DEFINE VARIABLES ######

#  Make a list of all RGI first order regions
reg_lst = geo_df['GLACIER_REGION_CODE'].unique().tolist()
reg_lst.remove('SAN')
reg_lst= reg_lst + ['SA1','SA2']

# # Test only Iceland
# reg_lst= ['ISL']

#  Make a list of the full period of interest 1915 to present
yr_lst = list(range(yr_ini, yr_end + 1))
# print(yr_lst)

min_year_geo_obs = 0 # start year of the period of interest, only geodetic from this year on will be considered, 0 if to use the start of the anomaly period
min_lenght_geo= 5 # minimum lenght in years of the geodetic observations accounted for anomaly calibration
run = 'spt_anom'

# all_wgms_id_lst = geo_df['WGMS_ID'].unique().tolist()
# print('Nb glaciers with geodetic obs C3S 2020: '+str(len(all_wgms_id_lst)))
# exit()

############################################################################################################################

###### DATA CRUNCHING: Slice glaciological and geodetic dataframe for a region of interest ######
# reg_lst= ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']
# reg_lst= ['ACN', 'ASC', 'RUA', 'CAU', 'ASW', 'ASE', 'ANT', 'SA2','ALA', 'ACS', 'SJM', 'SCA', 'ASN']
reg_lst= ['WNA', 'ISL', 'NZL', 'SA1', 'TRP', 'CEU', 'GRL']

for region in reg_lst:
    start_time = time.time()
    # region='TRP'
    print('working on region, ', region)

    # # create regional directory for regional glaciers OCE
    # out_reg_dir= os.path.join(path, 'out_data_'+fog_version+'_review', str(region) + '_oce_by_gla')
    # if not os.path.exists(out_reg_dir):
    #     os.mkdir(out_reg_dir)

    ## create regional OCE and sigma OCE empty dataframes, including 3 error sources dataframes and the full error
    reg_oce_df = pd.DataFrame(index=yr_lst)
    reg_oce_df.index.name = 'YEAR'
    reg_sig_dh_oce_df = pd.DataFrame(index=yr_lst)
    reg_sig_dh_oce_df.index.name = 'YEAR'
    reg_sig_rho_oce_df = pd.DataFrame(index=yr_lst)
    reg_sig_rho_oce_df.index.name = 'YEAR'
    reg_sig_anom_oce_df = pd.DataFrame(index=yr_lst)
    reg_sig_anom_oce_df.index.name = 'YEAR'

    reg_sig_oce_df = pd.DataFrame(index=yr_lst)
    reg_sig_oce_df.index.name = 'YEAR'

    ## number crunching: select geodetic data for glacier region group
    if region == 'SA1':
        geo_reg_df = geo_df.loc[(geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-01')]
        # print(geo_reg_df)
        # exit()

    elif region == 'SA2':
        geo_reg_df = geo_df.loc[(geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-02')]
        # print(geo_reg_df)
        # exit()

    else:
        geo_reg_df = geo_df.loc[(geo_df['GLACIER_REGION_CODE'] == str(region))]

    geo_reg_df['sigma_elevation_chg'] = geo_reg_df['sigma_elevation_chg'].fillna(geo_reg_df['sigma_elevation_chg'].mean())

    ## create a list of wgms_ids belonging to the region group
    reg_wgms_id_lst= geo_reg_df['WGMS_ID'].unique().tolist()
    # print(reg_wgms_id_lst)
    # print('Nb glaciers in region ' + str(region) + ' with geodetic obs C3S 2020: ' + str(len(reg_wgms_id_lst)))

    # # read regional anomaly data and uncertainties from csv files into dataframe
    all_CE_files = [f for f in os.listdir(path_spt_anom) if f.endswith('.csv')]
    reg_spt_anom_name = fnmatch.filter(all_CE_files, region + '*.csv')

    reg_spt_anom_file = os.path.join(path_spt_anom, reg_spt_anom_name[0])
    reg_spt_anom_df = pd.read_csv(reg_spt_anom_file, encoding='utf-8', delimiter=',', header=0, index_col='YEAR')

    reg_spt_anom_err_file = os.path.join(path_spt_anom, reg_spt_anom_name[1])
    reg_spt_anom_err_df = pd.read_csv(reg_spt_anom_err_file, encoding='utf-8', delimiter=',', header=0, index_col='YEAR')

    ############################################################################################################################

    ###### CALCULATING OCE: Loop through all glaciers in the region with available geodetic estimates ######

    for fog_id in reg_wgms_id_lst:
        # fog_id=491
        print('working on region, ', region, '- glacier Id, ', fog_id)

        # # create individual glacier directory
        # out_gla_dir = os.path.join(out_reg_dir, 'fog_Id_' + str(fog_id) + '_oce')
        # if not os.path.exists(out_gla_dir):
        #     os.mkdir(out_gla_dir)

        id_spt_anom_df = reg_spt_anom_df[[str(fog_id)]]
        id_spt_anom_err_df = reg_spt_anom_err_df[[str(fog_id)]]

        # # Define period of the complete anomaly series
        val = id_spt_anom_df[str(fog_id)].loc[id_spt_anom_df.first_valid_index()]

        if min_year_geo_obs == 0:
            if id_spt_anom_df.loc[id_spt_anom_df[str(fog_id)] == val].index[0] > 2000: ## For anomalies startig after 2000, use 2000 in order to use the geodetic estimates that start 2000
                min_year = 2000
            else:
                min_year = id_spt_anom_df.loc[id_spt_anom_df[str(fog_id)] == val].index[0]
        else:
            if id_spt_anom_df.loc[id_spt_anom_df[str(fog_id)] == val].index[0] > 2000:
                min_year = 2000
            else:
                min_year = min_year_geo_obs

        max_year = id_spt_anom_df.index.max()

        # # create geodetic mass balance series and geodetic Dataframe for selected glacier
        geo_mb_gla_df = geo_reg_df.loc[(geo_reg_df['WGMS_ID'] == fog_id)]

        # # Select geodetic estimates inside the period of interest and longer than min_lenght_geo

        geo_mb_gla_df = geo_mb_gla_df[['WGMS_ID', 'ini_date', 'fin_date', 'mb_chg_rate', 'sigma_tot_mb_chg', 'sigma_elevation_chg', 'elevation_chg_rate']]
        geo_ind_gla_sel_df = geo_mb_gla_df.loc[(geo_mb_gla_df['ini_date'] >= min_year - 2) & (geo_mb_gla_df['fin_date'] <= max_year + 1)]

        #create empty dataframes for calibrated series, calibrated series uncertainty, sigma geodetic uncertainty and distance to observation period
        cal_series_df = pd.DataFrame(index=yr_lst)
        cal_series_df.index.name='YEAR'

        sig_dh_cal_series_df = pd.DataFrame(index=yr_lst)
        sig_dh_cal_series_df.index.name='YEAR'

        sig_rho_cal_series_df = pd.DataFrame(index=yr_lst)
        sig_rho_cal_series_df.index.name='YEAR'

        sig_anom_cal_series_df = pd.DataFrame(index=yr_lst)
        sig_anom_cal_series_df.index.name = 'YEAR'

        sigma_geo_df = pd.DataFrame(index=yr_lst)
        sigma_geo_df.index.name= 'YEAR'

        dist_geo_df = pd.DataFrame(index=yr_lst)
        dist_geo_df.index.name= 'YEAR'
        dist_geo_df = dist_geo_df.reset_index()

        ###### Calculate the calibrated series #####
        for index, row in geo_ind_gla_sel_df.iterrows():
            if (int(row['fin_date'])-int(row['ini_date'])) >= min_lenght_geo:
                ref_period_geo_obs = range(int(row['ini_date']), int(row['fin_date']))
                ref_anom_period_df = id_spt_anom_df.loc[id_spt_anom_df.index.isin(ref_period_geo_obs)]
                ### -------here there is a problem with the dates !!!! need to correct geodetic values to hydrological years
                avg_ref_anom = ref_anom_period_df.mean()
                ref_anom= id_spt_anom_df - avg_ref_anom
                # print(ref_anom)

                cal_val = row['mb_chg_rate'] + ref_anom
                cal_series_df['serie_'+str(index)] = cal_val[str(fog_id)]

                # Three uncertainty sources: elevation change, density conversion, and anomaly

                # We save them in the same unit of mass change, multiplying dh by density
                # 1. Error in mass change from dh error sources
                sig_dh_cal_series_df['sig_serie_'+str(index)] = row['sigma_elevation_chg'] * 0.85
                # 2. Error in mass change from density error sources
                sig_rho_cal_series_df['sig_serie_'+str(index)] = np.abs(row['elevation_chg_rate']) * 0.06
                # 3. Error in mass change from anomalies
                sig_anom_cal_series_df['sig_serie_'+str(index)] = id_spt_anom_err_df

                # # Create geodetic estimate uncertainty dataframe
                sigma_geo_df['serie_' + str(index)]=row['sigma_tot_mb_chg']
                i_date=int(row['ini_date'])
                f_date=int(row['fin_date'])

                # # Create Distance to geodetic observation period dataframe
                dist_geo_df['serie_' + str(index)]= dist_geo_df['YEAR'].apply(lambda row: dis_fil(row, i_date, f_date))
            else:
                pass

        # if cal_series_df.empty == True:
        #     print('No calibrated series for glacier ' + str(wgms_id))

        if cal_series_df.empty == True:
            continue
            # os.rmdir(out_gla_dir)

        ###### Apply weights to calculate the mean calibrated series #####

        ## Calculate weight related to geodetic estimate uncertainty
        if cal_series_df.empty == False:

            if math.isnan(geo_ind_gla_sel_df['sigma_tot_mb_chg'].max()) == True:
                fill_sigma = 1.0
            else:
                fill_sigma = geo_ind_gla_sel_df['sigma_tot_mb_chg'].max()

            # weight_dir = out_gla_dir + 'wgmsId_' + str(wgms_id) + '_weights\\'
            # if not os.path.exists(weight_dir):
            #     os.mkdir(weight_dir)

            sigma_geo_df.fillna(fill_sigma, inplace=True)
            # sigma_geo_df.to_csv(weight_dir + 'Sigma_series_WGMS_ID_' + str(wgms_id) + '.csv', index=False)
            sigma_ratio_df=sigma_geo_df.apply(lambda x: (1 / x))
            wgt1_sigma_df=sigma_ratio_df.div(sigma_ratio_df.sum(axis=1), axis=0) # pass to percentage
            # wgt1_sigma_df.to_csv(weight_dir + 'Weight_percent_Sigma_series_WGMS_ID_' +str(wgms_id)+ '.csv', index=False)

            ## Calculate weight related to distance to the geodetic estimate survey period
            p=2 ## P value for inverse distance weighting
            dist_geo_df=dist_geo_df.set_index('YEAR')
            # dist_geo_df.to_csv(weight_dir + 'Distance_series_WGMS_ID_' + str(wgms_id) + '.csv', index=False)
            inv_dist_df=dist_geo_df.apply(lambda x: (1 / x) ** p)
            wgt2_dist_df=inv_dist_df.div(inv_dist_df.sum(axis=1), axis=0) # pass to percentage
            # wgt2_dist_df.to_csv(weight_dir + 'Weight_percent_Distance_series_WGMS_ID_' +str(wgms_id)+ '.csv', index=False)

            ## Calculate weight related to uncertaity and distance combined
            W1_W2_comb_df=wgt1_sigma_df.add(wgt2_dist_df)

            ##### Calculate MEANS of calibrated series: Artihmetic, Weight_1, Weight_2, Weight_combined #####

            cal_mean_df = pd.DataFrame(index=yr_lst)
            cal_mean_df.index.name='YEAR'
            # print(cal_series_df)

            ### Apply the weights to the calibrated series
            cal_series_W1_df = cal_series_df.mul(wgt1_sigma_df)
            cal_series_W2_df = cal_series_df.mul(wgt2_dist_df)
            cal_series_W1_W2_df=(cal_series_df.mul(W1_W2_comb_df))/2

            ## calibrated series means
            cal_mean_df['MEAN']=cal_series_df.mean(axis=1)
            cal_mean_df['MEAN_sigma_W']=cal_series_W1_df.sum(axis=1, min_count=1)
            cal_mean_df['MEAN_dist_W']=cal_series_W2_df.sum(axis=1, min_count=1)
            cal_mean_df['MEAN_combined_W']=cal_series_W1_W2_df.sum(axis=1, min_count=1)

            # # Plot the different means of the calibrated series
            # # print(cal_mean_df)
            # fig=cal_mean_df.plot()
            # plt.savefig(out_gla_dir + 'wgmsId_' + str(fog_id) + '_weights\\Fig_diff_mean_cal_series_glacier_id_' + str(fog_id) + '_' + region + '_Geo_+' + str(min_lenght_geo) + 'years.png')
            # plt.close()
            # exit()

            ## cumulative series of the different means
            cal_mean_cum_df = pd.DataFrame(index=yr_lst)
            cal_mean_cum_df.index.name='YEAR'

            cal_mean_cum_df['Cum_MEAN']=cal_mean_df['MEAN'].cumsum()
            cal_mean_cum_df['Cum_MEAN_sigma_W']=cal_mean_df['MEAN_sigma_W'].cumsum(skipna=True)
            cal_mean_cum_df['Cum_MEAN_dist_W']=cal_mean_df['MEAN_dist_W'].cumsum(skipna=True)
            cal_mean_cum_df['Cum_MEAN_combined_W']=cal_mean_df['MEAN_combined_W'].cumsum(skipna=True)

            ## Plot the cumulative values from the different means of the calibrated series
            # # print(cal_mean_cum_df)
            # fig=cal_mean_cum_df.plot()
            # plt.savefig(out_gla_dir + 'wgmsId_' + str(wgms_id) + '_weights\\Fig_cum_series_diff_mean_glacier_id_' + str(wgms_id) + '_Geo_+' + str(min_lenght_geo) + 'years.png')
            # plt.close()

            ############################################################################################################################

            # ### Calculate an save individual glacier OCE uncertainty

            ## ROMAIN: THIS ONE SHOULD NOT BE REQUIRED ANYMORE
            # nb_cal = cal_series_df.count(axis=1, numeric_only=True)
            # nb_cal = nb_cal.replace(0, np.nan)
            #
            # all_cal_yrs = cal_series_df.dropna()
            # cal_series_std = all_cal_yrs.std(axis=1).mean()
            # cal_series_var_err = 1.96 * cal_series_std / np.sqrt(nb_cal)  # Std-Error(Stdev/sqrt(N)) Error
            # print(cal_series_var_err)

            # sig_cal_sqsum = sig_cal_series_df.pow(2).sum(axis=1)
            # sig_cal_series_sum = np.sqrt(sig_cal_sqsum) / len(cal_series_df.columns)
            # print(sig_cal_mean)

            # print(sig_cal_series_sum)
            # if cal_series_df.empty == False:


            # We take the mean error for each source, which is fully justified
            # (they will be correlated for the same glacier)
            reg_sig_dh_oce_df[fog_id] = sig_dh_cal_series_df.mean(axis=1)
            reg_sig_rho_oce_df[fog_id] = sig_rho_cal_series_df.mean(axis=1)
            reg_sig_anom_oce_df[fog_id] = sig_anom_cal_series_df.mean(axis=1)

            # Total error
            reg_sig_oce_df[fog_id] = np.sqrt(sig_anom_cal_series_df.mean(axis=1) ** 2 +
                                             sig_rho_cal_series_df.mean(axis=1) ** 2 +
                                             sig_dh_cal_series_df.mean(axis=1) ** 2)


            # sig_oce.to_csv(out_gla_dir + 'sigma_CE_fog_id_' + str(fog_id) + '_' + region + '.csv')
            #

            # TODO: INES I HAD TO PUT THIS OUT OF THE IF LOOP FOR PLOT BELOW, OTHERWISE "ISL_regional_CEs.csv" WAS EMPTY
            #  OR IT WAS DOING AN EXIT()

            oce_df = pd.DataFrame()
            oce_df['weighted_MEAN'] = cal_series_W1_W2_df.sum(axis=1, min_count=1)
            oce_df['normal_MEAN'] = cal_series_df.mean(axis=1)
            oce_df = oce_df.rename(columns={'weighted_MEAN': fog_id})
            oce_df = oce_df[fog_id]
            reg_oce_df[fog_id] = oce_df
            # print(oce_df)

            # ## Plot and save individual glacier OCE
            # if len(cal_series_df.columns) >= 1:
            #     plot_gla_oce(cal_series_df, geo_ind_gla_sel_df, fog_id, min_year, max_year, min_lenght_geo, region, run, out_gla_dir)
            #     plot_gla_oce_and_unc(cal_series_df, oce_df, geo_ind_gla_sel_df, reg_sig_oce_df, fog_id, min_year, max_year, min_lenght_geo, region, run, out_gla_dir)


                # exit()
    # print(reg_oce_df)
    # print(reg_sig_oce_df)
        read_time4 = time.time()
        print("--- %s seconds ---" % (read_time4 - start_time))


    ### Save regional OCEs
    reg_oce_df.to_csv(os.path.join(out_dir, region + '_regional_CEs.csv'))
    reg_sig_dh_oce_df.to_csv(os.path.join(out_dir, region + '_regional_sigma_dh_CEs.csv'))
    reg_sig_rho_oce_df.to_csv(os.path.join(out_dir, region + '_regional_sigma_rho_CEs.csv'))
    reg_sig_anom_oce_df.to_csv(os.path.join(out_dir, region + '_regional_sigma_anom_CEs.csv'))
    # exit()

print('.........................................................................................')
print('"The End"')
print('.........................................................................................')
exit()
