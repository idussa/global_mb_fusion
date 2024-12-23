"""
Calculate the regional mass loss

calc_reg_mass_loss.py

Author: idussa
Date: Feb 2021
Last changes: Feb 2021

Scripted for Python 3.7

Description:
This script reads glacier-wide mass balance data edited from WGMS FoG database
and regional glacier anomalies produced by calc_regional_anomalies_and_error.py
and provides the observational consensus estimate for every individual glacier
with available geodetic observations WGMS Id

Input:  C3S_GLACIER_DATA_20200824.csv
        OCE_files_by_region\\
        (UTF-8 encoding)

Return: tbd.svg
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions_ggmc import *
import scipy  # usage: scipy.stats.funct()

rgi_region= {'ACN' : 'Arctic Canada (North)', 'WNA' : 'Western Canada & US', 'ALA' : 'Alaska', 'ACS' : 'Arctic Canada (South)', 'TRP' : 'Low Latitudes', 'SCA' : 'Scandinavia',
             'SJM' : 'Svalbard', 'CEU' : 'Central Europe', 'CAU' : 'Caucasus & Middle East', 'ASC' : 'Central Asia', 'ASN' : 'North Asia', 'ASE' : 'South Asia (East)',
             'NZL' : 'New Zealand', 'ASW' : 'South Asia (West)', 'GRL' : 'Greenland Periphery', 'ANT' : 'Antarctic & Subantarctic', 'ISL' : 'Iceland', 'RUA' : 'Russian Arctic',
             'SA1' : 'Southern Andes (Patagonia)', 'SA2' : 'Southern Andes (Central)', 'SAN' : 'Southern Andes'}

rgi_code= {'ALA' : '1', 'WNA' : '2', 'ACN' : '3', 'ACS' : '4', 'GRL' : '5', 'ISL' : '6', 'SJM' : '7', 'SCA' : '8', 'RUA' : '9', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SAN' : '17', 'SA1' : '17', 'SA2' : '17', 'NZL' : '18', 'ANT' : '19'}

rgi_code_hug= {'ALA' : '01', 'WNA' : '02', 'ACN' : '03', 'ACS' : '04', 'GRL' : '05', 'ISL' : '06', 'SJM' : '07', 'SCA' : '08', 'RUA' : '09', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SAN' : '17', 'SA1' : '17', 'SA2' : '17', 'NZL' : '18', 'ANT' : '19'}

# reg_lst= ['ALA','WNA', 'ACN', 'ACS', 'ISL', 'SJM', 'SCA', 'CEU', 'CAU', 'TRP', 'SA1', 'SA2', 'NZL']
reg_lst= ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']

color = {  'ALA':	'#b9762a',
           'WNA':   '#6e628f',
           'ACN':	'#aeba7e',
           'ACS':	'#53835f',
           'GRL':	'#511b2e',
           'ISL':	'#da6769',
           'SJM':	'#983910',
           'SCA':	'#0d4a20',
           'RUA':	'#4a2555',
           'ASN':   '#d8b54d',
           'CEU':   '#5260ff',
           'CAU':   '#cce09a',
           'ASC':   '#0aaf8e',
           'ASW':   '#124542',
           'ASE':   '#7a3d8d',
           'TRP':   '#de989b',
           'SA1':   '#1c6e69',
           'SA2':   '#1c6e69',
           'NZL':   '#3c1422',
           'ANT':	'#d2d275'}

##########################################
##########################################
"""main code"""
##########################################
##########################################

#################################################################################################
##    Define input datasets
#################################################################################################

path = os.path.dirname(os.path.abspath(__file__))
path_proj = 'C:\\Users\\idussail2\\Documents\\PROJECTS\\G3P_project\\'
fog_version = '2024-01'

in_ba_file = path_proj + 'codes\\mb_data_crunching_local\\4.2_Kriging_regional_mass_balance\\out_data_'+fog_version+'_review\\Regional_B_series_AreaWeighted.csv'
in_sig_ba_file = path_proj + 'codes\\mb_data_crunching_local\\4.2_Kriging_regional_mass_balance\\out_data_'+fog_version+'_review\\Regional_B_series_uncertainty.csv'
path_oce = path_proj + 'codes\\mb_data_crunching_local\\3.3_Kriging_global_CE_spatial_anomaly\\out_data_'+fog_version+'_review\\OCE_files_by_region\\'# path to regional OCE files

in_area_zemp_file = path + '\\in_data\\Regional_area_change_Zemp_for_spt_CEs.csv' # Regional Area changes from Zempe et al 2019
in_data_area = path + '\\in_data\\_RGI_All_ID_Area.csv '
in_data_zemp = path + '\\in_data\\zemp_etal_regional_series\\'
in_data_hug = path + '\\in_data\\hugonnet_et_al_regional_series\\'

#################################################################################################
##    Define parameters
#################################################################################################

# period to calculate the cumulative mass loss
ini_yr_full_obs = 1976
fin_yr_obs = 2023

# PoR= list(range(1976, 2023+1)) #
# PoR = list(range(2000, 2019+1)) # common period with Hugonnet et al. 2021
PoR= list(range(1976, 2016+1)) # common period with Zemp et al. 2019
# PoR = list(range(2014, 2023+1)) #

PoR_full = list(range(ini_yr_full_obs, fin_yr_obs+1)) #

S_ocean = 362.5 * 10**6 # Cogley et al 2012

sig_area = 0.05 # Paul et al 2015

# Define Output parameters
out_dir = path + '\\out_data_'+fog_version+'_review\\'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
#
plt_year_min = 1950 # starting year for Regional series plot
plt_year_max = 2025 # end year for Regional series plot

# axes = 'eq_axes' # all region with same Y axes, visualizes best the contribution between regions
axes = 'tight' # fits Y axes to each region, visualizes best the temporal variability of each region

#################################################################################################
##    READ input files
#################################################################################################

ba_df = pd.read_csv(in_ba_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
sig_ba_df = pd.read_csv(in_sig_ba_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')

# ba_df = ba_df / 1000
# sig_ba_df = sig_ba_df / 1000

reg_area_zemp_df = pd.read_csv(in_area_zemp_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
id_area_df = pd.read_csv(in_data_area, encoding='utf-8', delimiter=',', header=0)

############################################################################################################################

###### Calculate total glacier mass loss by region ######

cols = ['region', 'area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]', 'area_mean_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [km2]',
        'percentage_area_obs' ,'DM [Gt yr-1]', 'sigma_DM [Gt yr-1]', 'CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]', 'sigma_CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]',
        'B [mwe yr-1]', 'sigma_B [mwe yr-1]',  'SLE [mm yr-1]', 'sigma_SLE [mm yr-1]', 'zemp_DM [Gt yr-1]', 'zemp_sigma_DM [Gt yr-1]',
        'zemp_CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]']


glob_cum_df = pd.DataFrame(index=reg_lst, columns=cols)

Reg_DM_df = pd.DataFrame()
Reg_sig_DM_df = pd.DataFrame()

for region in reg_lst:
    # region = 'ISL'
    print('working in region: ', region)

    out_DM_series = out_dir +'regional_mass_loss_series\\'
    if not os.path.exists(out_DM_series):
        os.mkdir(out_DM_series)

    in_oce_file = path_oce + region + '_regional_CEs.csv'
    oce_df = pd.read_csv(in_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')

    rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == region)].set_index('WGMS_ID')

    if region == 'SA1':
        rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == 'SAN')].set_index('WGMS_ID')
        rgi_area_df= rgi_area_df.loc[rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-01']
    #
    if region == 'SA2':
        rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == 'SAN')].set_index('WGMS_ID')
        rgi_area_df= rgi_area_df.loc[rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-02']

    ## select wgms_ids belonging to the region group
    wgms_id_lst = oce_df.columns.to_list()
    wgms_id_lst = [int(i) for i in wgms_id_lst]
    # print(len(wgms_id_lst))

    id_lst=[]
    for id in wgms_id_lst:
        if id in rgi_area_df.index:
            id_lst.append(id)
        else:
            pass
    ## Calculate total area of observed glaciers presenting an area value in FoG

    nb_gla_reg = len(rgi_area_df)
    # print(nb_gla_reg)
    gla_obs_df = rgi_area_df.loc[id_lst]

    tot_area_rgi_reg = rgi_area_df['AREA'].sum()
    tot_area_obs_reg = gla_obs_df['AREA'].sum()

    nb_gla_obs = len(gla_obs_df)
    ba_mwe = ba_df[region]
    ba_kmwe = ba_mwe/10**3

    sig_ba_mwe = sig_ba_df[region]
    sig_ba_kmwe = sig_ba_mwe/10**3

    area = reg_area_zemp_df[region]
    # area_reg_for_weight = round(area.loc[2022])

    dm_Gt = ba_kmwe * area

    sig_dm_sum = np.sqrt((sig_ba_kmwe/ba_kmwe)**2 + sig_area**2)
    sig_dm = np.abs(dm_Gt) * sig_dm_sum

    reg_file = pd.DataFrame()
    reg_file['Aw_mwe'] = ba_df[region]
    reg_file['sig_tot_mwe'] = sig_ba_df[region]
    reg_file['area_tot_km2'] = reg_area_zemp_df[region]
    reg_file['DM_Gt'] = dm_Gt
    reg_file['sig_tot_DM'] = sig_dm

    reg_file.to_csv(out_DM_series + 'results_region_' + rgi_code[region] + '_' + region + '_' + rgi_region[region] + '.csv')

    Reg_DM_df[region] = dm_Gt
    Reg_sig_DM_df[region] = sig_dm

####### Calculate cumulative mass loss for PoR ###################################

    mean_area = reg_file['area_tot_km2'].loc[(reg_file.index.isin(PoR))].mean()
    mean_area_full = reg_file['area_tot_km2'].loc[(reg_file.index.isin(PoR_full))].mean()

    B_mwe_yr = reg_file['Aw_mwe'].loc[(reg_file.index.isin(PoR))].mean()
    sigma_B_mwe_yr = np.sqrt(reg_file['sig_tot_mwe'].loc[(reg_file.index.isin(PoR))].pow(2).sum() / len(PoR))

    # CUM_B_mwe = reg_file['Aw_mwe'].loc[(reg_file.index.isin(PoR))].cumsum()
    # CUM_B_mwe = CUM_B_mwe.loc[max(PoR)]
    # print(CUM_B_mwe)
    # exit()

    DM_Gt_yr = reg_file['DM_Gt'].loc[(reg_file.index.isin(PoR))].sum() / len(PoR)
    CUM_DM_Gt = reg_file['DM_Gt'].loc[(reg_file.index.isin(PoR))].sum()

    sigma_DM_Gt_yr = np.sqrt(reg_file['sig_tot_DM'].loc[(reg_file.index.isin(PoR))].pow(2).sum() / len(PoR))  # maybe not divided by the lenght of the period!
    sigma_cum = reg_file['sig_tot_DM'].pow(2).cumsum().pow(0.5)
    sigma_CUM_DM_Gt = sigma_cum.loc[2023]

    # sig = glob_df['sigma_DM [Gt]'].pow(2).cumsum()
    # glob_df['sigma_DM_cum [Gt]'] = sig.pow(0.5)

    per_obs = nb_gla_obs * 100 / nb_gla_reg
    per_area = tot_area_obs_reg * 100 / tot_area_rgi_reg
    SLE = (-DM_Gt_yr / S_ocean) * 10**6
    Sigma_SLE = (sigma_DM_Gt_yr / S_ocean) * 10**6

    # for index, row in glob_cum_df.iterrows():
    #     if index == region:
    #         row['region']= rgi_region[region]
    #         row['area_mean [km2]'] = "{:.0f}".format(mean_area)
    #         row['percentage_gla_obs'] = round(per_obs)
    #         row['DM [Gt yr-1]'] = "{:.2f}".format(DM_Gt_yr)
    #         row['sigma_DM [Gt yr-1]'] = "{:.2f}".format(sigma_DM_Gt_yr)
    #         row['B [mwe yr-1]'] = "{:.2f}".format(B_mwe_yr)
    #         row['sigma_B [mwe yr-1]'] = "{:.2f}".format(sigma_B_mwe_yr)
    #         row['CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]'] = "{:.2f}".format(CUM_DM_Gt)
    #         # row['CUM_B_'+str(min(PoR))+'_'+str(max(PoR))+' [mwe]'] = "{:.2f}".format(CUM_B_mwe)
    #         row['SLE [mm yr-1]'] = "{:.3f}".format(SLE)
    #         row['sigma_SLE [mm yr-1]'] = "{:.3f}".format(Sigma_SLE)

    ############# PLOT regional mass loss

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # plot zero balance line
    ax1.axhline(color='Black', linewidth=0.75)

    # plot zemp regional B
    if region in ['SA1', 'SA2']:
        in_zemp_df = in_data_zemp + 'Zemp_etal_results_region_' + rgi_code[region] + '_SAN.csv'
    else:
        in_zemp_df = in_data_zemp + 'Zemp_etal_results_region_' + rgi_code[region] + '_' + region + '.csv'


    zemp_df = pd.read_csv(in_zemp_df, encoding='utf-8', delimiter=',', header=26, index_col='Year')
    zemp_2019_df = zemp_df[[' INT_Gt', ' sig_Int_Gt', ' sig_Total_Gt']]
    zemp_2019_df.index.name = 'Year'
    zemp_2019_df.columns = ['MB_Gt', 'sig_Int_Gt', 'MB_sigma_Gt']

    zemp_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum() / len(PoR)
    # zemp_1976_2016 = zemp_2019_df['MB_Gt'].sum() / len(PoR)
    zemp_sig_1976_2016 = np.sqrt(zemp_2019_df['MB_sigma_Gt'].loc[zemp_2019_df.index.isin(PoR)].pow(2).sum() / len(PoR))
    zemp_cum_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum()

    zemp_2019_x_df = zemp_2019_df.index
    zemp_2019_y_df = zemp_2019_df['MB_Gt']
    zemp_2019_y_uci_df = zemp_2019_df['MB_Gt'] + zemp_2019_df['MB_sigma_Gt']
    zemp_2019_y_lci_df = zemp_2019_df['MB_Gt'] - zemp_2019_df['MB_sigma_Gt']
    ax1.plot(zemp_2019_x_df.values, zemp_2019_y_df.values, color=color['ASC'], label='Zemp et al. (2019)')
    ax1.fill_between(zemp_2019_x_df.values, zemp_2019_y_uci_df.values, zemp_2019_y_lci_df.values, color=color['ASC'], alpha=0.3)

    # plt hugonnet
    if region in ['SA1', 'SA2']:
        in_hug_df = in_data_hug + 'dh_17_rgi60_reg_rates.csv'
    else:
        in_hug_df = in_data_hug + 'dh_' + rgi_code_hug[region] + '_rgi60_reg_rates.csv'

    hug_df = pd.read_csv(in_hug_df, encoding='utf-8', delimiter=',', header=0)
    hug_df = hug_df.head(20)
    hug_df['Year'] = range(2000,2020,1)
    hug_2021_df = hug_df.set_index('Year')

    hug_2021_x_df = hug_2021_df.index
    hug_2021_y_df = hug_2021_df['dmdt']
    hug_2021_y_uci_df = hug_2021_df['dmdt'] + hug_2021_df['err_dmdt']
    hug_2021_y_lci_df = hug_2021_df['dmdt'] - hug_2021_df['err_dmdt']
    ax1.plot(hug_2021_x_df.values, hug_2021_y_df.values, color='Orange', label='Hugonnet et al. (2021)')
    ax1.fill_between(hug_2021_x_df.values, hug_2021_y_uci_df.values, hug_2021_y_lci_df.values, color='Orange', alpha=0.5)

    # prepare and plot in_data
    in_data1_x_df = reg_file.index
    in_data1_y_df = reg_file['DM_Gt']
    in_data1_y_uci_df = reg_file['DM_Gt'] + reg_file['sig_tot_DM']
    in_data1_y_lci_df = reg_file['DM_Gt'] - reg_file['sig_tot_DM']

    ax1.plot(in_data1_x_df.values, in_data1_y_df.values, color='black', label='This study', alpha=0.6)
    ax1.fill_between(in_data1_x_df.values, in_data1_y_uci_df.values, in_data1_y_lci_df.values, color='grey', alpha=0.5)

    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim([(ymin + 0.5*ymin), (ymax + 0.5*ymax)])
    ax1.set_xlim([plt_year_min, plt_year_max])

    # Add axis labels, title, legend, text
    # ax1.legend(loc='upper right', fontsize='small')
    # ax1.set_xlabel('Year', fontsize='large')
    # ax1.set_ylabel('Annual mass change (Gt)', fontsize='large')

    # get y-range from first y-axis and convert to units of second y-axis
    y_min_mslre = gt2slr(ymin)
    y_max_mslre = gt2slr(ymax)
    ax2.set_ylim(y_min_mslre, y_max_mslre)
    # ax2.set_ylabel('Sea-level equivalent (mm)', fontsize='large')

    ymin2, ymax2 = ax1.get_ylim()
    ax1.text(1952, (ymin2 - ymin2*0.05), rgi_code[region] + '-' + rgi_region[region], size=22, weight=600)
    ax1.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # optimize layout
    fig.tight_layout()
    plt.rcParams["figure.dpi"] = 300

    out_plot_dir = out_dir + 'plot_reg_mass_loss_Gt_'+str(plt_year_min)+'-'+str(plt_year_max)+'_'+axes+'\\'
    if not os.path.exists(out_plot_dir):
        os.mkdir(out_plot_dir)

    out_fig = out_plot_dir + region + '_dM_series_Gt_'+axes+'.svg'
    plt.savefig(out_fig)
    print('Plot saved as {}.'.format(out_fig))
    plt.close()

    for index, row in glob_cum_df.iterrows():
        if index == region:
            row['region']= rgi_region[region]
            row['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'] = "{:.0f}".format(mean_area_full)
            row['area_mean_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [km2]'] = "{:.0f}".format(mean_area)
            row['percentage_area_obs'] = per_area
            row['DM [Gt yr-1]'] = "{:.2f}".format(DM_Gt_yr)
            row['sigma_DM [Gt yr-1]'] = "{:.2f}".format(sigma_DM_Gt_yr)
            row['CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]'] = "{:.2f}".format(CUM_DM_Gt)
            row['sigma_CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]'] = "{:.2f}".format(sigma_CUM_DM_Gt)
            row['B [mwe yr-1]'] = "{:.2f}".format(B_mwe_yr)
            row['sigma_B [mwe yr-1]'] = "{:.2f}".format(sigma_B_mwe_yr)
            row['SLE [mm yr-1]'] = "{:.3f}".format(SLE)
            row['sigma_SLE [mm yr-1]'] = "{:.3f}".format(Sigma_SLE)
            row['zemp_DM [Gt yr-1]'] = "{:.2f}".format(zemp_1976_2016)
            row['zemp_sigma_DM [Gt yr-1]'] = "{:.2f}".format(zemp_sig_1976_2016)
            row['zemp_CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]'] = "{:.2f}".format(zemp_cum_1976_2016)
    # exit()

# #### For ploting SA1 and SA2 together
#
# SAN_in = out_dir+'SAN_DM_series.csv'
# SAN_df = pd.read_csv(SAN_in, encoding='utf-8', delimiter=',', header=0, index_col='YEAR')
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# # plot zero balance line
# ax1.axhline(color='Black', linewidth=0.75)
#
# # plot zemp regional B
# in_zemp_df = in_data_zemp + 'Zemp_etal_results_region_' + rgi_code['SAN'] + '_SAN.csv'
#
# zemp_df = pd.read_csv(in_zemp_df, encoding='utf-8', delimiter=',', header=26, index_col='Year')
# zemp_2019_df = zemp_df[[' INT_Gt', ' sig_Int_Gt', ' sig_Total_Gt']]
# zemp_2019_df.index.name = 'Year'
# zemp_2019_df.columns = ['MB_Gt', 'sig_Int_Gt', 'MB_sigma_Gt']
#
# zemp_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum() / len(PoR)
# zemp_sig_1976_2016 = np.sqrt(zemp_2019_df['MB_sigma_Gt'].loc[zemp_2019_df.index.isin(PoR)].pow(2).sum() / len(PoR))
# zemp_cum_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum()
#
# zemp_2019_x_df = zemp_2019_df.index
# zemp_2019_y_df = zemp_2019_df['MB_Gt']
# zemp_2019_y_uci_df = zemp_2019_df['MB_Gt'] + zemp_2019_df['MB_sigma_Gt']
# zemp_2019_y_lci_df = zemp_2019_df['MB_Gt'] - zemp_2019_df['MB_sigma_Gt']
# ax1.plot(zemp_2019_x_df.values, zemp_2019_y_df.values, color=color['ASC'], label='Zemp et al. (2019)')
# ax1.fill_between(zemp_2019_x_df.values, zemp_2019_y_uci_df.values, zemp_2019_y_lci_df.values, color=color['ASC'], alpha=0.3)
#
# # plt hugonnet
# in_hug_df = in_data_hug + 'dh_' + rgi_code_hug['SAN'] + '_rgi60_reg_rates.csv'
#
# hug_df = pd.read_csv(in_hug_df, encoding='utf-8', delimiter=',', header=0)
# hug_df = hug_df.head(20)
# hug_df['Year'] = range(2000,2020,1)
# hug_2021_df = hug_df.set_index('Year')
#
# hug_2021_x_df = hug_2021_df.index
# hug_2021_y_df = hug_2021_df['dmdt']
# hug_2021_y_uci_df = hug_2021_df['dmdt'] + hug_2021_df['err_dmdt']
# hug_2021_y_lci_df = hug_2021_df['dmdt'] - hug_2021_df['err_dmdt']
# ax1.plot(hug_2021_x_df.values, hug_2021_y_df.values, color='Orange', label='Hugonnet et al. (2021)')
# ax1.fill_between(hug_2021_x_df.values, hug_2021_y_uci_df.values, hug_2021_y_lci_df.values, color='Orange', alpha=0.5)
#
# # prepare and plot in_data
# in_data1_x_df = SAN_df.index
# in_data1_y_df = SAN_df['SAN']
# in_data1_y_uci_df = SAN_df['SAN'] + SAN_df['sigma_SAN']
# in_data1_y_lci_df = SAN_df['SAN'] - SAN_df['sigma_SAN']
#
# ax1.plot(in_data1_x_df, in_data1_y_df, color='black', label='This study', alpha=0.6)
# ax1.fill_between(in_data1_x_df, in_data1_y_uci_df, in_data1_y_lci_df, color='grey', alpha=0.5)
#
# ymin, ymax = ax1.get_ylim()
# ax1.set_ylim([(ymin + 0.5*ymin), (ymax + 0.5*ymax)])
# ax1.set_xlim([plt_year_min, plt_year_max])
#
# # Add axis labels, title, legend, text
# # ax1.legend(loc='upper right', fontsize='small')
# # ax1.set_xlabel('Year', fontsize='large')
# # ax1.set_ylabel('Annual mass change (Gt)', fontsize='large')
#
# # get y-range from first y-axis and convert to units of second y-axis
# y_min_mslre = gt2slr(ymin)
# y_max_mslre = gt2slr(ymax)
# ax2.set_ylim(y_min_mslre, y_max_mslre)
# # ax2.set_ylabel('Sea-level equivalent (mm)', fontsize='large')
#
# ymin2, ymax2 = ax1.get_ylim()
# ax1.text(1952, (ymin2 - ymin2*0.05), rgi_code['SAN'] + '-' + rgi_region['SAN'], size=22, weight=600)
# ax1.tick_params(labelsize=16)
# ax2.tick_params(labelsize=16)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.spines['left'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.spines['left'].set_visible(False)
#
# # optimize layout
# fig.tight_layout()
# plt.rcParams["figure.dpi"] = 300
#
# out_plot_dir = out_dir + 'plot_reg_mass_loss_Gt_'+str(plt_year_min)+'-'+str(plt_year_max)+'_'+axes+'\\'
# if not os.path.exists(out_plot_dir):
#     os.mkdir(out_plot_dir)
#
# out_fig = out_plot_dir + 'SAN_dM_series_Gt_'+axes+'.svg'
# plt.savefig(out_fig)
# print('Plot saved as {}.'.format(out_fig))
# plt.close()
# exit()

#################################################################################################
########## Calculate global glacier mass balance and mass loss series ############################

# if PoR = PoR_full:
area_per_region = glob_cum_df['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'].astype(float)
world_area = glob_cum_df['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'].astype(float).sum()
rel_area_per_region = area_per_region / world_area
rel_area_dict = rel_area_per_region.to_dict()

# glob_ba_df2 = ba_df.loc[(ba_df.index >= ini_yr_full_obs)].mean(axis=1).rename('B [m w.e.]')
# glob_sig_ba2 = sig_ba_df.loc[(sig_ba_df.index >= ini_yr_full_obs) ].pow(2).sum(axis=1) /20
# glob_sig_ba_df2 = np.sqrt(glob_sig_ba2).rename('sigma B [m w.e.]')

ba_df0 = pd.DataFrame()
for region in ba_df.columns:
    ba_df0[region] = ba_df[region] * rel_area_dict[region]

# print(ba_df)
# print(ba_df0)

glob_ba_df = ba_df0.loc[(ba_df0.index >= ini_yr_full_obs)].sum(axis=1).rename('B [m w.e.]')

sig_ba_df0 = pd.DataFrame()
for region in sig_ba_df.columns:
    sig_ba_df0[region] = sig_ba_df[region] * rel_area_dict[region]

glob_sig_ba = sig_ba_df0.loc[(sig_ba_df0.index >= ini_yr_full_obs)].sum(axis=1).rename('sigma B [m w.e.]')
glob_sig_ba_df = np.sqrt(glob_sig_ba.pow(2))
# print(glob_sig_ba_df)

glob_DM_df = Reg_DM_df.loc[(Reg_DM_df.index >= ini_yr_full_obs) & (Reg_DM_df.index <= fin_yr_obs)].sum(axis=1).rename('DM [Gt]')
glob_sig_DM = Reg_sig_DM_df.loc[(Reg_DM_df.index >= ini_yr_full_obs) & (Reg_DM_df.index <= fin_yr_obs)].pow(2).sum(axis=1)
glob_sig_DM_df = np.sqrt(glob_sig_DM).rename('sigma_DM [Gt]')

glob_slr_df = (-glob_DM_df/ S_ocean) * 10 ** 6
glob_slr_df = glob_slr_df.rename('SLE [mm]')
glob_sig_slr_df = (glob_sig_DM_df / S_ocean) * 10 ** 6
glob_sig_slr_df = glob_sig_slr_df.rename('sigma_SLE [mm]')

glob_df = pd.concat([glob_ba_df, glob_sig_ba_df, glob_DM_df, glob_sig_DM_df, glob_slr_df, glob_sig_slr_df ], axis=1)
# print(glob_df)
# exit()

Reg_DM_cum_df = Reg_DM_df.loc[Reg_DM_df.index >= 1976]
Reg_DM_cum_df = Reg_DM_cum_df.cumsum()

Reg_B_cum_df = ba_df.loc[ba_df.index >= 1976]
Reg_B_cum_df = Reg_B_cum_df.cumsum()
Reg_B_cum_df.to_csv(out_dir + 'Cumulative_Regional_Bmwe_series.csv')

Reg_DM_df.to_csv(out_dir + 'Regional_DM_series.csv')
Reg_DM_cum_df.to_csv(out_dir + 'Cumulative_Regional_DM_series.csv')
Reg_sig_DM_df.to_csv(out_dir + 'Regional_DM_series_uncertainty.csv')
plt.show()

#### 1. MASS LOSS
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# plot zero balance line
ax1.axhline(color='Black', linewidth=0.75)
ax1.set_xlim([1955, 2025])
ax1.set_ylim([-650, 400])
y1, y2 = ax1.get_ylim()

in_zemp_df = in_data_zemp + 'zemp2019_global_estimates_corr_nopretxt_1962-2016.csv'
zemp_df = pd.read_csv(in_zemp_df, encoding='utf-8', delimiter=',', header=0, index_col='Year')
zemp_2019_df = zemp_df[[' INT_Gt', ' sig_Total_Gt']]
zemp_2019_df.columns = ['MB_Gt', 'MB_sigma_Gt']
zemp_2019_x_df = zemp_2019_df.index
zemp_2019_y_df = zemp_2019_df['MB_Gt']
zemp_2019_y_uci_df = zemp_2019_df['MB_Gt'] + zemp_2019_df['MB_sigma_Gt']
zemp_2019_y_lci_df = zemp_2019_df['MB_Gt'] - zemp_2019_df['MB_sigma_Gt']
ax1.plot(zemp_2019_x_df.values, zemp_2019_y_df.values, color=color['ASC'], label='Zemp et al. (2019)')
ax1.fill_between(zemp_2019_x_df.values, zemp_2019_y_uci_df.values, zemp_2019_y_lci_df.values, color=color['ASC'], alpha=0.3)

in_hug_glob_df = in_data_hug + 'hugonnet2021_global_estimates_sourcedatafig1_2000-2019.csv'
hug_glob_df = pd.read_csv(in_hug_glob_df, encoding='utf-8', delimiter=',', header=0, index_col='Year')
hug_2021_x_df = hug_glob_df.index
hug_2021_y_df = hug_glob_df['dMdt_Gtyr-1']
hug_2021_y_uci_df = hug_glob_df['dMdt_Gtyr-1'] + hug_glob_df['dMdt_sigma_Gtyr-1']
hug_2021_y_lci_df = hug_glob_df['dMdt_Gtyr-1'] - hug_glob_df['dMdt_sigma_Gtyr-1']
ax1.plot(hug_2021_x_df.values, hug_2021_y_df.values, color='Orange', label='Hugonnet et al. (2021)')
ax1.fill_between(hug_2021_x_df.values, hug_2021_y_uci_df.values, hug_2021_y_lci_df.values, color='Orange', alpha=0.5)

in_data1_x_df = glob_df.index
in_data1_y_df = glob_df['DM [Gt]']
in_data1_y_uci_df = glob_df['DM [Gt]'] + glob_df['sigma_DM [Gt]']
in_data1_y_lci_df = glob_df['DM [Gt]'] - glob_df['sigma_DM [Gt]']

ax1.plot(in_data1_x_df.values, in_data1_y_df.values, color='black', label='This study', alpha=0.6)
ax1.fill_between(in_data1_x_df.values, in_data1_y_uci_df.values, in_data1_y_lci_df.values, color='grey', alpha=0.5)

# Add axis labels, title, legend, text
ax1.legend(loc='upper right', fontsize='large', frameon=False)
ax1.set_xlabel('Year', size=14, weight=600)
ax1.set_ylabel('Annual mass change (Gt)', size=14, weight=600)

y_min_mslre = gt2slr(y1)
y_max_mslre = gt2slr(y2)
ax2.set_ylim(y_min_mslre, y_max_mslre)
ax2.set_ylabel('Sea-level equivalent (mm)', size=14, weight=600)

ax1.text(1958, -630, 'Global', size=22, weight=600)
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# optimize layout
fig.tight_layout()
plt.rcParams["figure.dpi"] = 300

out_fig = out_dir + 'plot_global_DM_series_Gt_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + '.svg'
plt.savefig(out_fig)
plt.close()

#### 2. MASS BALANCE

ax = glob_df['B [m w.e.]'].plot(color='black', linewidth=2)
# plt.title(' Global glacier mass balance (m w.e.)', fontsize=16)
plt.xlim([1975, 2025], )

plt.ylim([-1.5, 1.0])

plt.axhline(0, color='Grey', linewidth=0.8)
# ax.set_ylabel('Mass balance (m w.e.)', fontsize=18)
# plt.xlabel('Year', fontsize='medium')

ax.tick_params(labelsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.text(1975, -1.45, 'Global', size=22, weight=600)
plt.fill_between(glob_df['B [m w.e.]'].index, glob_df['B [m w.e.]'] + glob_df['sigma B [m w.e.]'],
                 glob_df['B [m w.e.]'] - glob_df['sigma B [m w.e.]'], color='silver', alpha=0.5, linewidth=0)

out_fig = out_dir + 'plot_global_B_series_mwe_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + '.svg'
plt.savefig(out_fig)
plt.close()

########## Calculate global glacier cumulative mass loss ############################

glob_area_full = glob_cum_df['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'].astype(float).sum()
glob_area_mean = glob_cum_df['area_mean_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [km2]'].astype(float).sum()
glob_per_obs = glob_cum_df['percentage_area_obs'].mean()
glob_DM_yr = glob_cum_df['DM [Gt yr-1]'].astype(float).sum()
glob_sig_DM_yr = "{:.1f}".format((np.sqrt(glob_cum_df['sigma_DM [Gt yr-1]'].astype(float).pow(2).sum()/20)))
glob_DM_CUM = glob_cum_df['CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]'].astype(float).sum()
glob_SLE_yr = glob_cum_df['SLE [mm yr-1]'].astype(float).sum()
glob_sig_SLE_yr = "{:.3f}".format((np.sqrt(glob_cum_df['sigma_SLE [mm yr-1]'].astype(float).pow(2).sum())))

glob_lst = ['GLOBAL', '' , glob_area_full, glob_area_mean , glob_per_obs, glob_DM_yr, glob_sig_DM_yr, glob_DM_CUM, '', '', '', glob_SLE_yr, glob_sig_SLE_yr, '', '', '' ]
glob_cum_df = glob_cum_df.reset_index()
glob_cum_df.loc[len(glob_cum_df.index)] = glob_lst
glob_cum_df = glob_cum_df.rename(columns={'index':'region_code'})

glob_df['DM_cum [Gt]']= glob_df['DM [Gt]'].cumsum()

sig = glob_df['sigma_DM [Gt]'].pow(2).cumsum()
glob_df['sigma_DM_cum [Gt]'] = sig.pow(0.5)

glob_df['B_cum [m w.e.]']= glob_df['B [m w.e.]'].cumsum()

glob_df['SLE_cum [Gt]']= glob_df['SLE [mm]'].cumsum()
sig_sle = glob_df['sigma_SLE [mm]'].pow(2).cumsum()
glob_df['sigma_SLE_cum [mm]'] = sig_sle.pow(0.5)

glob_df.to_csv(out_dir + 'Global_DM_series_year_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + '.csv')
glob_cum_df.to_csv(out_dir + 'Cum_DM_Gt_per_region_PoR_'+str(min(PoR))+'_'+str(max(PoR))+'.csv', index=False)

#### 1.2 PLOT CUM GLOB MASS LOSS
fig2, ax1 = plt.subplots()
ax2 = ax1.twinx()

# plot zero balance line
ax1.axhline(color='Black', linewidth=0.75)
ax1.set_xlim([1970, 2025])
ax1.set_ylim([-9000, 1000])
y1, y2 = ax1.get_ylim()

x_df = glob_df.index
y_df = glob_df['DM_cum [Gt]']
y_uci_df = glob_df['DM_cum [Gt]'] + glob_df['sigma_DM_cum [Gt]']
y_lci_df = glob_df['DM_cum [Gt]'] - glob_df['sigma_DM_cum [Gt]']

ax1.plot(x_df.values, y_df.values, color='black', alpha=0.6, linewidth=2)
ax1.fill_between(x_df.values, y_uci_df.values, y_lci_df.values, color='silver', alpha=0.5)

y_min_mslre = gt2slr(y1)
y_max_mslre = gt2slr(y2)
ax2.set_ylim(y_min_mslre, y_max_mslre)
ax2.set_ylabel('Sea-level equivalent (mm)', fontsize='x-large')

# Add axis labels, title, legend, text
# ax1.legend(loc='upper right', fontsize='medium')
ax1.set_xlabel('Year', fontsize='x-large')
ax1.set_ylabel('Cumulative mass change (Gt)', fontsize='x-large')

ax1.text(1971, -7800, 'Global', color= 'Black', size=36, weight=600)
ax1.text(1971, -8600, 'Glacier mass-change', color= 'silver', size=20, weight=600)
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# optimize layout
fig2.tight_layout()
plt.rcParams["figure.dpi"] = 300

out_fig = out_dir + 'plot_CUM_global_DM_series_Gt_' + str(ini_yr_full_obs) + '-' + str(fin_yr_obs) + '.svg'
plt.savefig(out_fig)
plt.close()


# set output format
file_type = '.svg'

# define name of figure
out_fig = out_dir + 'plot_global_DM_series_Gt_bars_' + str(ini_yr_full_obs) + '-' + str(fin_yr_obs) + file_type

fig3 = plt.figure()
ax = fig3.add_subplot(111)

# plot zero balance line
ax.axhline(0, color='Grey', linewidth=1)

glob_df = glob_df.reset_index()
# plot geodetic mass change trends
for index, row in glob_df.iterrows():
    color = 'b' if row['DM [Gt]'] > 0 else 'r'
    x1 = row['YEAR'] - 1
    x2 = x1 + 1
    y1 = row['DM [Gt]']
    # print(x1)
    # print(x2)
    # exit()
    ax.fill([x1, x1, x2, x2], [0, y1, y1, 0], color, alpha=0.5)
    ax.plot([x1, x1, x2, x2], [0, y1, y1, 0], color, linewidth=0.5, solid_capstyle='butt')


ax.set_xlabel('Year', fontsize='x-large')
ax.set_ylabel('Annual mass change (Gt)', fontsize='x-large')

ax.text(1971, -560, 'Global', color= 'Black', size=36, weight=600)
ax.text(1971, -620, 'Glacier mass-change', color= 'silver', size=20, weight=600)
ax.tick_params(labelsize=14)
ax.tick_params(labelsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)



# ax.text(1952, -3.2, 'B$_{geo}$', fontsize=24)

# save plot
plt.xlim(1970, 2025)
plt.ylim((-650, 150))
# plt.legend(loc=3)
plt.rcParams["figure.dpi"] = 300
plt.tight_layout()
plt.savefig(out_fig)
print('Plot saved as {}.'.format(out_fig))
# plt.show()

# clear plot
plt.close()



### Mass balance mwe bars
# set output format
file_type = '.svg'

# define name of figure
out_fig = out_dir + 'plot_global_B_series_mwe_bars_' + str(ini_yr_full_obs) + '-' + str(fin_yr_obs) + file_type

fig3 = plt.figure(figsize=(6.4, 2))
ax = fig3.add_subplot(111)

# plot zero balance line
ax.axhline(0, color='Grey', linewidth=1)

glob_df = glob_df.reset_index()
# plot geodetic mass change trends
for index, row in glob_df.iterrows():
    color = 'silver' if row['B [m w.e.]'] > 0 else '#907ab7'
    x1 = row['YEAR'] - 1
    x2 = x1 + 1
    y1 = row['B [m w.e.]']
    # print(x1)
    # print(x2)
    # exit()
    ax.fill([x1, x1, x2, x2], [0, y1, y1, 0], color, alpha=0.5)
    ax.plot([x1, x1, x2, x2], [0, y1, y1, 0], color, linewidth=0.5, solid_capstyle='butt')


ax.set_xlabel('Year', fontsize='x-large')
ax.set_ylabel('Annual specific mass change (m w.e.)', fontsize='x-large')

ax.text(1971, -0.96, 'Global', color= 'Black', size=36, weight=600)
# ax.text(1971, -620, 'Glacier mass-change', color= 'silver', size=20, weight=600)
ax.tick_params(labelsize=14)
ax.tick_params(labelsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)



# ax.text(1952, -3.2, 'B$_{geo}$', fontsize=24)

# save plot
plt.xlim(1970, 2025)
plt.ylim((-1, 0.2))
# plt.legend(loc=3)
plt.rcParams["figure.dpi"] = 300
plt.tight_layout()
plt.savefig(out_fig)
print('Plot saved as {}.'.format(out_fig))
# plt.show()

# clear plot
plt.close()

print('.........................................................................................')
print('"The End"')
print('.........................................................................................')
exit()


