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



import math
import numpy as np
import os, sys, shutil, csv
import matplotlib.pyplot as plt
import pandas as pd
import time
from functions_ggmc import *
from gcdistance import *
pd.options.mode.chained_assignment = None  # default='warn'
import scipy  # usage: scipy.stats.funct()

##########################################
##########################################
"""main code"""
##########################################
##########################################
##### DEFINE VARIABLES ######

fog_version = '2024-01'

# Define reference period to calculate anomalies
year_ini = 2011
year_fin = 2020
reference_period = range(year_ini, year_fin + 1)
# print(list(reference_period))


max_glac_anom = 5 # maximum number of closer individual glacier anomalies used to calculate the glacier temporal variability
min_glac_anom = 3 # minimum number of closer individual glacier anomalies to calculate the glacier temporal variability, if less anomalies are available, regional anomaly is used
d_thresh_lst = [60, 120, 250, 500, 1000] # search distances (km) for finding close mass balance anomalies
max_d = 1000 # maximum distance (km) allowed for finding close mass balance anomalies, if no anomalies are found, regional anomaly is used

plt_year_min = 1970 # starting year for Regional series plot
plt_year_max = 2023 # end year for Regional series plot

axes = 'eq_axes' # all region with same Y axes, visualizes best the contribution between regions
# axes = 'tight' # fits Y axes to each region, visualizes best the temporal variability of each region

# Define input
##### 1. PATH TO FILES ######
start_time = time.time()

path = os.path.dirname(os.path.abspath(__file__))

out_dir = path + '\\out_data_'+fog_version+'_review\\'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# create directory for regional glaciers anomalies
out_reg_dir= out_dir + 'MEAN_spatial_gla_anom_ref_'+str(year_ini)+'-'+str(year_fin)+'\\'
if not os.path.exists(out_reg_dir):
    os.mkdir(out_reg_dir)

out_anom_dir= out_dir + 'LOOKUP_spatial_and_reg_ids_ref_'+str(year_ini)+'-'+str(year_fin)+'\\'
if not os.path.exists(out_anom_dir):
    os.mkdir(out_anom_dir)

out_long_dir= out_dir + 'LONG-NORM_spatial_gla_anom_ref_' + str(year_ini) + '-' + str(year_fin) + '\\'
if not os.path.exists(out_long_dir):
    os.mkdir(out_long_dir)

##### 2.1 READ MASS BALANCE DATA ######

# read FoG file with global annual and seasonal mass-balance data
in_data_gla = path + '\\in_data\\fog-'+fog_version+'\\fog_bw-bs-ba_'+fog_version+'.csv'
input_gla_df = pd.read_csv(in_data_gla, delimiter=',', header=0)

### create mass-balance data csv if it has not been produced before

# create unique list of glacier ids and years with data
all_fog_gla_id_lst = input_gla_df['WGMS_ID'].unique().tolist()
yr_lst = list(range(1915, max(input_gla_df['YEAR']+1), 1))
# print(max(input_gla_df['YEAR']))
# print(yr_lst)

reg_lst = input_gla_df['GLACIER_REGION_CODE'].unique().tolist()
reg_lst.remove('SAN')
reg_lst= reg_lst + ['SA1','SA2'] # Separate Andes in two regions:

ba_file = path + '\\in_data\\fog-'+fog_version+'\\fog_' + fog_version+ '_ba.csv'
ba_unc_file = path + '\\in_data\\fog-'+fog_version+'\\fog_' + fog_version+ '_ba_unc.csv'

# ba_df = create_mb_dataframe(input_gla_df, all_fog_gla_id_lst, yr_lst, 'ANNUAL_BALANCE')
# ba_df.to_csv(ba_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')
# ba_unc_df = create_mb_dataframe(input_gla_df, all_fog_gla_id_lst, yr_lst, 'ANNUAL_BALANCE_UNC')
# ba_unc_df.to_csv(ba_unc_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')
# exit()

# read FoG file with global annual mass-balance data
ba_df = pd.read_csv(ba_file, delimiter=',', header=0, index_col=0)
ba_df.columns = ba_df.columns.map(int)  # make columns names great again

### Add missing years to Urumqi glacier fog_id 853
file= path + '\\in_data\\urumqi_missing_years.csv'
df = pd.read_csv(file, delimiter=',', header=0, index_col=0)
df.columns = df.columns.map(int)  # make columns names great again
ba_df = ba_df.fillna(df)

ba_unc_df = pd.read_csv(ba_unc_file, delimiter=',', header=0, index_col=0)
ba_unc_df.columns = ba_unc_df.columns.map(int)  # make columns names great again

in_gla_coord = path + '\\in_data\\fog-'+fog_version+'\\FOG_coord_'+fog_version+'.csv'
coord_gla_df= pd.read_csv(in_gla_coord, encoding='latin1', delimiter=',', header=0, index_col='WGMS_ID').sort_index()

##### 2.2 READ GEODETIC DATA ######

# read FoG file with global geodetic data
in_data_geo = path + '\\in_data\\fog-'+fog_version+'\\_FOG_GEO_MASS_BALANCE_DATA_'+fog_version+'.csv'
input_geo_df= pd.read_csv(in_data_geo, encoding='latin1', delimiter=',', header=0, index_col='WGMS_ID').sort_index()
input_geo_df.reset_index(inplace=True)
# print(geo_df)
# exit()

all_fog_geo_id_lst = input_geo_df['WGMS_ID'].unique().tolist()
# print('Nb glaciers with geodetic obs C3S 2022: '+str(len(all_wgms_id_lst)))
# exit()

read_time = time.time()
print("--- %s seconds ---" % (read_time - start_time))
############################################################################################################################
# print(reg_lst)
# exit()
# reg_lst = ['ASC', 'GRL', 'ASW', 'CAU', 'RUA', 'ISL', 'ASE', 'ANT', 'NZL', 'SA1', 'SA2']
#
# for region in reg_lst:
#     # region= 'CEU'
#     print('working on region, ', region)
#
#     out_csv = out_anom_dir + 'readme_final_ids_for_mean_anom_' + str(region) + '.csv'
#     with open(out_csv, 'w', encoding='utf-8') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',', lineterminator = '\n')
#         writer.writerow(['REGION', 'WGMS_ID', 'LIST_IDS_MEAN_ANOM'])
#
#         ## create empty dataframes for spatial anomalies and uncertainties
#         spt_anom_df = pd.DataFrame(index=yr_lst)
#         spt_anom_df.index.name = 'YEAR'
#         spt_anom_lst = []
#
#         spt_anom_err_df = pd.DataFrame(index=yr_lst)
#         spt_anom_err_df.index.name = 'YEAR'
#         sig_spt_anom_lst = []
#
#         ## number crunching: SELECT GEODETIC DATA FOR GLACIER REGION GROUP
#
#         if region == 'SA1':
#             reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-01')]
#         elif region == 'SA2':
#             reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-02')]
#         else:
#             reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_REGION_CODE'] == str(region))]
#
#         ## create a list of fog_ids with geodetic data for the region group
#         reg_fog_geo_id_lst= reg_geo_df['WGMS_ID'].unique().tolist()
#
#         ############################################################################################################################
#         ###### 3. CALCULATING SPATIAL ANOMALIES: Loop through all glaciers in the region ######
#
#         for fog_id in reg_fog_geo_id_lst:
#             # fog_id = 491
#             print('working on glacier Id, ', fog_id)
#
#             ## SELECT MASS BALANCE DATA FOR GLACIER REGION GROUP
#             ## create list of glacier mass balance series ids possible to calculate the glacier temporal variabiity or anomaly
#             ## remove or add neighbouring glacier mass balance series
#
#             if region == 'ASN': # add Urumqui, remove Hamagury yuki, add
#                 add_id_lst = [853, 817]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
#                 rem_id = 897  # Hamagury yuki (ASN)
#                 rem_id_lst2 = [897, 1511, 1512]  # Hamagury yuki (ASN), Urumqi East and west branches (ASC)
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ALA')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASC')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst2)].index)
#                 glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'] == rem_id].index)
#                 # print(list(glac['WGMS_ID'].unique().tolist()))
#                 # exit()
#
#             if region == 'ASE':
#                 add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
#                 rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASC')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
#                 glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#
#             if region == 'ASC':
#                 rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASE')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
#                 glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'].isin(rem_id_lst)].index)
#
#             if region == 'ASW':
#                 add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
#                 rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASC')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASE')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
#                 glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#
#             if region == 'CEU':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = glac
#
#             if region == 'SA1':
#                 rem_id_lst = [3902, 3903, 3904, 3905, 1344, 3972]  # keep Martial Este only
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
#                 glac_reg = glac
#                 # print(list(glac['WGMS_ID'].unique().tolist()))
#                 # exit()
#
#             if region == 'SA2':
#                 rem_id_lst = [3902, 3903, 3904, 3905, 2000, 3972] #  keep Echaurren Norte only
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
#                 glac_reg = glac
#
#             if region == 'NZL':
#                 add_id_lst = [2000]  # Martial Este (SAN-01)
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = glac
#
#             if region == 'ANT':
#                 rem_id_lst = [878, 3973]  # Dry valley glaciers
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
#                 glac_reg = glac
#
#             if region == 'RUA':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'SJM') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = glac
#
#             if region == 'SJM':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = glac
#
#             if region == 'ALA':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'WNA') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#
#             if region == 'WNA':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ALA'), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#
#             if region == 'TRP':
#                 rem_id = 226  # Yanamarey
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac = glac.drop(glac[glac['WGMS_ID'] == rem_id].index)
#                 glac_reg = glac
#
#             if region == 'ACS':
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = glac
#
#             if region == 'ACN':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = glac
#
#             if region == 'GRL':
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#
#             if region == 'ISL':
#                 glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'GRL')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#
#             if region == 'SCA':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = glac
#
#             if region == 'CAU':
#                 glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'CEU'), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#                 glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
#
#             ## Find all possible individual glacier anomalies (with respect to reference period) for the given glacier id
#
#             ## number crunching:   select mass-balance data for glacier region groups
#             ba_glac_df = ba_df.loc[:, list(glac['WGMS_ID'].unique().tolist())]
#             glac_anom = calc_anomalies(ba_glac_df, reference_period, region)
#             # print(glac_anom)
#             # exit()
#
#             unc_glac_anom = calc_spt_anomalies_unc(glac_anom, ba_unc_df, glac_anom.columns.to_list() )
#
#             # FOR SA2 ONLY: if no uncertainty measurement use the regional annual mean uncertainty of the glaciological sample
#             if unc_glac_anom.isnull().sum().sum():
#                 for id in unc_glac_anom.columns.tolist():
#                     year_min = glac_anom[id].first_valid_index()
#                     yrs = list(range(1915, year_min))
#                     unc_glac_anom[id].fillna(np.nanmean(ba_unc_df), inplace=True)
#                     unc_glac_anom[id].mask(unc_glac_anom.index.isin(yrs), np.nan, inplace=True)
#             else:
#                 continue
#
#             ## Correct suspicious anomaly from Echaurren Norte by normalizing past period to present period amplitude.
#             if region == 'SA2':
#                 STD_ech_ok = glac_anom.loc[glac_anom.index.isin(list(range(2004, (2022 + 1))))].std()
#                 STD_ech_bad = glac_anom.loc[glac_anom.index.isin(list(range(1980, (1999 + 1))))].std()
#                 glac_anom_pres_ok = glac_anom.loc[glac_anom.index >= 2004]
#                 norm_past = glac_anom.loc[glac_anom.index.isin(list(range(1885, (2003 + 1))))] / STD_ech_bad
#                 glac_anom_past_new = (norm_past * STD_ech_ok).round(decimals=1)
#                 glac_anom = pd.concat([glac_anom_past_new, glac_anom_pres_ok], axis = 0)
#
#
#             glac_possible_lst = glac_anom.columns.to_list()
#
#             ## Find the closest glacier anomalies to the given glacier id: nb-to_anom defines the number of series selected
#             close_gla_df = coord_gla_df.loc[glac_possible_lst, :]
#
#             lat_id= coord_gla_df.loc[fog_id, 'LATITUDE']
#             lon_id = coord_gla_df.loc[fog_id, 'LONGITUDE']
#             lat_glac = close_gla_df['LATITUDE']
#             lon_glac= close_gla_df['LONGITUDE']
#
#             close_gla_df['distance (km) to id:' + str(fog_id)] = great_circle_distance(lat_id, lon_id, lat_glac, lon_glac)
#             close_gla_df = close_gla_df.sort_values(by=['distance (km) to id:' + str(fog_id)])
#             close_gla_df = close_gla_df.loc[(close_gla_df['distance (km) to id:' + str(fog_id)] < max_d)]
#             nb_gla_close = np.count_nonzero(close_gla_df['distance (km) to id:' + str(fog_id)] < max_d) ## count number of glacier anomalies found within 1000km to the fog_id glacier
#             # print(close_gla_df)
#             # print(nb_gla_close)
#             # exit()
#
#             ## Filter series for regional anomaly to use if no anomalies are found close to the glacier
#             ba_reg_glac_df = ba_df.loc[:, list(glac_reg['WGMS_ID'].unique().tolist())]
#             reg_glac_anom = calc_anomalies(ba_reg_glac_df, reference_period, region)
#
#             ## Correct suspicious anomaly from Echaurren Norte by normalizing past period to present period amplitude.
#             if region == 'SA2':
#                 STD_ech_ok = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(2004, (2020 + 1))))].std()
#                 STD_ech_bad = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1980, (1999 + 1))))].std()
#                 reg_glac_anom_pres_ok = reg_glac_anom.loc[reg_glac_anom.index >= 2000]
#                 norm_past = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1885, (1999 + 1))))] / STD_ech_bad
#                 reg_glac_anom_past_new = (norm_past * STD_ech_ok).round(decimals=1)
#                 reg_glac_anom = pd.concat([glac_anom_past_new, glac_anom_pres_ok], axis = 0)
#
#             ## select close anomalies for calculating the fog_id glacier anomaly
#             if nb_gla_close <= min_glac_anom:
#                 anoms_4_fog_id_df = reg_glac_anom
#                 spatial_id_fin_lst = reg_glac_anom.columns.to_list()
#
#             else:
#                 for d in d_thresh_lst:
#                     d_close_gla_df = close_gla_df[close_gla_df['distance (km) to id:' + str(fog_id)] <= d]
#                     if len(d_close_gla_df) <= min_glac_anom:
#                         continue
#                     if len(d_close_gla_df) > min_glac_anom:
#                         d_sel_close_gla_df = d_close_gla_df
#                     break
#
#                 spatial_id_sel_lst = d_sel_close_gla_df.index.to_list()
#
#                 ## Check if regional longer series are in the list, if not, add them
#                 if region in ['ALA', 'ACS', 'ACN', 'ASN','CAU','GRL','NZL','RUA','SA1','SA2','ANT']:
#                     spatial_id_fin_lst = spatial_id_sel_lst
#                 if region in ['ASE', 'ASC']:
#                     add_id_lst = [817,853] # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                         spatial_id_fin_lst = list(set(spatial_id_fin_lst))
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#                 if region == 'ASW':
#                     add_id_lst = [817] # Ts. Tuyuksuyskiy (ASC)
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#                 if region == 'CEU':
#                     add_id_lst = [356,491,408,635] # St. Sorlin, Hinteeisferner, Silvretta, Caresser (CEU)
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                         spatial_id_fin_lst = list(set(spatial_id_fin_lst))
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#                 if region == 'ISL':
#                     add_id_lst = [3089] # Hofsjokull N
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#                 if region in ['SJM', 'RUA']:
#                     add_id_lst = [291,292] # Midtre Lovenbreen, Austre Broeggerbreen (SJM)
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                         spatial_id_fin_lst = list(set(spatial_id_fin_lst))
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#                 if region == 'SCA':
#                     add_id_lst = [302,332] # Storbreen, Storglaciereren (SCA)
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                         spatial_id_fin_lst = list(set(spatial_id_fin_lst))
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#                 if region == 'TRP':
#                     add_id_lst = [26615] # Zongo
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#                 if region == 'WNA':
#                     add_id_lst = [205] # South cascade
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#
#                 if region == 'NZL':
#                     add_id_lst = [2000] # Martial Este (SAN-01)
#                     check = all(item in spatial_id_sel_lst for item in add_id_lst)
#                     if check is False:
#                         spatial_id_fin_lst = spatial_id_sel_lst + add_id_lst
#                     else:
#                         spatial_id_fin_lst = spatial_id_sel_lst
#
#             # print(spatial_id_fin_lst)
#             # exit()
#
#             close_gla_weights = coord_gla_df.loc[spatial_id_fin_lst, :]
#             lat_glac = close_gla_weights['LATITUDE']
#             lon_glac= close_gla_weights['LONGITUDE']
#
#             close_gla_weights['distance (km) to id:' + str(fog_id)] = great_circle_distance(lat_id, lon_id, lat_glac, lon_glac)
#             close_gla_weights = close_gla_weights.sort_values(by=['distance (km) to id:' + str(fog_id)])
#             close_gla_weights['inv_dist_weight'] = np.where((close_gla_weights['distance (km) to id:' + str(fog_id)] != 0), (1/ close_gla_weights['distance (km) to id:' + str(fog_id)])**1, 1 )
#             # close_gla_weights['dist_weight_unc'] = np.where((close_gla_weights['distance (km) to id:' + str(fog_id)] != 0), (close_gla_weights['distance (km) to id:' + str(fog_id)])**-(-1), 1 )
#
#
#             # glac_id_sel_lst = d_close_gla_df.index.to_list()[0:max_glac_anom] ## only if a maximum number of anomalies want to be used
#             anoms_4_fog_id_df = glac_anom[spatial_id_fin_lst]
#             unc_anoms_4_fog_id_df = unc_glac_anom[spatial_id_fin_lst]
#
#             idw_df = anoms_4_fog_id_df.copy()
#             idw_df = idw_df.notnull().astype('int')
#
#             # unc_w_df = anoms_4_fog_id_df.copy()
#             # unc_w_df = unc_w_df.notnull().astype('int')
#
#             for id in spatial_id_fin_lst:
#                 idw_weight= close_gla_weights.loc[id, 'inv_dist_weight']
#                 idw_df[id] = idw_df[id] * idw_weight
#                 idw_df[id].replace(0, np.nan, inplace=True)
#
#                 # unc_weight= close_gla_weights.loc[id, 'dist_weight_unc']
#                 # unc_w_df[id] = unc_w_df[id] * unc_weight
#                 # unc_w_df[id].replace(0, np.nan, inplace=True)
#
#             yr_sum_idw = idw_df.sum(axis=1)
#             yr_sum_idw.replace(0, np.nan, inplace=True)
#
#             # yr_sum_unc_w = unc_w_df.sum(axis=1)
#             # yr_sum_unc_w.replace(0, np.nan, inplace=True)
#
#             weight_anom_fog_id_df = (anoms_4_fog_id_df * idw_df).sum(axis=1)
#             weight_anom_fog_id_df.replace(0, np.nan, inplace=True)
#
#             weight_unc_anom_fog_id_df = (unc_anoms_4_fog_id_df * idw_df).sum(axis=1)
#             weight_unc_anom_fog_id_df.replace(0, np.nan, inplace=True)
#
#             ## Print list of glaciological series used to calculate the anomaly of every given glacier
#             writer.writerow([str(region), str(fog_id), spatial_id_fin_lst])
#
#             # ## Plot the anomalies used to calculate the mean anomaly
#             # ax = anoms_4_fog_id_df.plot(title='Temporal coverage Glaciological sample region ' + str(region) + '\n N = ' + str(
#             #     len(anoms_4_fog_id_df.columns)) + ' glaciers', fontsize='medium')
#             # ax.set_xlabel('Year', fontsize='medium')
#             # ax.set_xlim([1960, 2020])
#             # ax.set_ylabel('MB (mm w.e. yr-1)', fontsize='medium')
#             # plt.show()
#             #
#             # print(anoms_4_fog_id_df)
#             # exit()
#
#             ## CALCULATE:  mean anomaly for fog_id
#             ## if glacier has in situ measurements i.e. dist = 0 use the own glaciers anomaly
#
#             anom_fog_id_df = weight_anom_fog_id_df / yr_sum_idw
#             # print(anom_fog_id_df)
#             # exit()
#             anom_fog_id_df = pd.DataFrame(anom_fog_id_df, columns=[str(fog_id)])
#             anom_fog_id_df = anom_fog_id_df.loc[anom_fog_id_df.index >= 1915]
#
#             spt_anom_lst.append(anom_fog_id_df)
#
#             ## CALCULATE: Uncertainty for fog_id
#             mean_gla_mba_err = weight_unc_anom_fog_id_df / yr_sum_idw
#             mean_gla_mba_err_df = pd.DataFrame(mean_gla_mba_err, columns=[str(fog_id)])
#
#             nb_gla = pd.DataFrame(anoms_4_fog_id_df.count(axis=1, numeric_only = True), columns=[str(fog_id)])
#             nb_gla = nb_gla.replace(0, np.nan)
#             nb_gla = nb_gla.dropna()
#
#             anoms_std = pd.DataFrame(anoms_4_fog_id_df.std(axis=1), columns=[str(fog_id)])
#             anoms_std = anoms_std.mean()
#             # print(anoms_std)
#             # exit()
#
#             var_anoms_err_df = (anoms_std / np.sqrt(nb_gla))  # Std-Error(Stdev/sqrt(N)) Error
#
#             # fill nans with mean error
#             min_year = anom_fog_id_df.first_valid_index()
#             yrs = list(range(1885, min_year))
#             var_anoms_err_df.fillna(var_anoms_err_df.mean(), inplace=True)
#             var_anoms_err_df.loc[var_anoms_err_df.index.isin(yrs), [str(fog_id)]] = np.nan
#
#             sig_anom_df = np.sqrt(var_anoms_err_df.pow(2) + mean_gla_mba_err_df.pow(2))
#             sig_anom_df = round(sig_anom_df,2)
#             sig_anom_df = sig_anom_df.loc[sig_anom_df.index >= 1915]
#             # sig_anom_df = 1.96 * sig_anom_df
#             # print(sig_anom_df)
#             # exit()
#
#             sig_spt_anom_lst.append(sig_anom_df)
#             # print(sig_anom_df)
#             # exit()
#
#             # ## PLOT glacier anomalies and Regional mean
#             #
#             # fig = plt.figure()
#             # ax = fig.add_subplot(111)
#             #
#             # out_plot_path = out_dir +'plot_reg_anomaly_ref_period_' + str(year_ini) + '-' + str(year_fin) + '_final\\'
#             # if not os.path.exists(out_plot_path):
#             #     os.mkdir(out_plot_path)
#             # # print(anoms_4_fog_id_df)
#             # # exit()
#             # anoms_4_fog_id_df = anoms_4_fog_id_df/1000
#             #
#             # if len(anoms_4_fog_id_df.columns) > 1:
#             #     plot_anom_df = anoms_4_fog_id_df.mul(1)
#             #     plot_anom_df['UNC'] =  sig_anom_df/1000 * 2
#             #     plot_anom_df['MEAN'] = anom_fog_id_df/1000
#             #
#             #     ax = anoms_4_fog_id_df.plot(color='grey', linewidth=0.5, legend=False)
#             #     ax.set_ylim([-3, 3])
#             #     ax.axhline(0, color='Grey', linewidth=1)
#             #     ax.set_ylabel('\u03B2 (m w.e.)', size=18, weight=600)
#             #     ax.set_xlabel('Year', size=18, weight=600)
#             #     ax.text(1995, 3, 'N = ' + str(len(anoms_4_fog_id_df.columns)) + ' glaciers', size=14, weight=600)
#             #
#             #     plot_anom_df['MEAN'].plot(ax=ax, color='black', alpha=0.7, linewidth=2)
#             #     plt.fill_between(plot_anom_df.index, plot_anom_df['MEAN'] + plot_anom_df['UNC'],
#             #                      plot_anom_df['MEAN'] - plot_anom_df['UNC'], color='grey', alpha=0.3, linewidth=0)
#             #
#             #     # save plot
#             #     ax.tick_params(labelsize=18)
#             #     ax.spines['right'].set_visible(False)
#             #     ax.spines['top'].set_visible(False)
#             #     ax.spines['left'].set_visible(False)
#             #     ax.set_xlim([1950, 2023])
#             #     plt.xticks(np.arange(1960, 2023, 20))
#             #
#             #     out_fig = out_plot_path + 'Anomaly_and_UNC_for_id_' + str(fog_id) + '_ref_period_' + str(year_ini) + '-' + str(
#             #         year_fin) + '_' + fog_version + '.png'
#             #
#             #     fig.tight_layout()
#             #     plt.savefig(out_fig, dpi=300)
#             #     print('Plot saved as {}.'.format(out_fig))
#             #     # reg_anom.plot()
#             #     plt.close()
#             # else:
#             #     print('................... Region with only one glacier anomaly:', region,
#             #           '............................')
#             # exit()
#     glac_anom.to_csv(out_anom_dir + region + '_all_SEL_gla_anomalies.csv')
#     reg_glac_anom.to_csv(out_anom_dir + region + '_all_reg_gla_anomalies.csv')
#     unc_glac_anom.to_csv(out_anom_dir + region + '_all_SEL_gla_anomalies_UNC.csv')
#
#     ### Save all glacier anomalies and uncertainties - exclude uncertainties from the SAN regions
#     spt_anom_df = pd.concat(spt_anom_lst, axis='columns')
#     spt_anom_df.to_csv(out_reg_dir + str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
#
#     if not (region in ['SA1', 'SA2']):
#         sig_spt_anom_df = pd.concat(sig_spt_anom_lst, axis='columns')
#         sig_spt_anom_df.to_csv(out_reg_dir + str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
#
#     print("--- %s seconds ---" % (time.time() - read_time))
#
#     ### Save glacier anomalies and uncertainties OK with long time periods
#     reg_ok_lst = ['ACS', 'ACN', 'ASW', 'ASE', 'ASC', 'ASN', 'ALA', 'SCA']
#     if region in reg_ok_lst:
#         spt_anom_df.to_csv(out_long_dir + str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
#         sig_spt_anom_df.to_csv(out_long_dir + str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
#     if region == 'SA2':
#         spt_anom_df.to_csv(out_long_dir + str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
#
# exit()
#
# ###############   SPECIAL CASE ANDES:
#
# sa1_anom_in = out_anom_dir + 'SA1_all_reg_gla_anomalies.csv'
# sa2_anom_in = out_anom_dir + 'SA2_all_reg_gla_anomalies.csv'
# sa1_anom_df = pd.read_csv(sa1_anom_in, delimiter=',', header=0, index_col=0)
# sa2_anom_df = pd.read_csv(sa2_anom_in, delimiter=',', header=0, index_col=0)
# san_anom_df = pd.concat([sa1_anom_df,sa2_anom_df], axis='columns')
#
# nb_gla = pd.DataFrame(san_anom_df.count(axis=1, numeric_only=True), columns=[' '])
# nb_gla = nb_gla.replace(0, np.nan)
# # nb_gla = nb_gla.dropna()
#
# anoms_std = pd.DataFrame(san_anom_df.std(axis=1), columns=[' '])
# anoms_std = anoms_std.mean()
#
# san_var_err = (anoms_std / np.sqrt(nb_gla))  # Std-Error(Stdev/sqrt(N)) Error
#
# sa1_sig_in = out_anom_dir + 'SA1_all_SEL_gla_anomalies_UNC.csv'
# sa2_sig_in = out_anom_dir + 'SA2_all_SEL_gla_anomalies_UNC.csv'
# sa1_sig_df = pd.read_csv(sa1_sig_in, delimiter=',', header=0, index_col=0)
# sa2_sig_df = pd.read_csv(sa2_sig_in, delimiter=',', header=0, index_col=0)
# san_sig_df = pd.concat([sa1_sig_df,sa2_sig_df], axis='columns')
# san_mean_mba_err = pd.DataFrame(san_sig_df.mean(axis=1), columns=[' '])
#
# san_anom_df['SA1_unc'] = np.sqrt(san_mean_mba_err.pow(2) + san_var_err.pow(2))
# san_anom_df['SA2_unc'] = np.sqrt(san_mean_mba_err.pow(2) + san_var_err.pow(2))
# # print(san_anom_df)
# # exit()
#
# ## fill nans with mean error
# min_year = sa2_anom_df.first_valid_index()
# yrs = list(range(1915, min_year))
# san_anom_df['SA2_unc'].fillna(san_anom_df['SA2_unc'].mean(), inplace=True)
# san_anom_df.loc[san_anom_df.index.isin(yrs), ['SA2_unc']] = np.nan
# # san_anom_df = 1.96 * san_anom_df
#
# for region in ['SA1', 'SA2']:
#     print('working in region: ', region)
#     san_anom_df[region+'_unc'].to_csv(out_anom_dir + region + '_all_SEL_gla_anomalies_UNC_recalc.csv')
#
#     san_err_df = pd.DataFrame(index=yr_lst)
#     san_err_df.index.name = 'YEAR'
#     sig_san_lst= []
#
#     if region == 'SA1':
#         reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-01')]
#     elif region == 'SA2':
#         reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-02')]
#     else:
#         reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_REGION_CODE'] == str(region))]
#
#     ## create a list of fog_ids with geodetic data for the region group
#     reg_fog_geo_id_lst= reg_geo_df['WGMS_ID'].unique().tolist()
#     # print(reg_fog_geo_id_lst)
#
#     for fog_id in reg_fog_geo_id_lst:
#         print('working on glacier id:', fog_id)
#         san_err_df[fog_id] = san_anom_df[region+'_unc']
#         # print(san_err_df[fog_id])
#         sig_san_lst.append(san_err_df[fog_id])
#         # exit()
#
#     sig_spt_anom_df = pd.concat(sig_san_lst, axis='columns')
#     sig_spt_anom_df.to_csv(out_reg_dir + str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
#
#     if region == 'SA2':
#         sig_spt_anom_df.to_csv(out_long_dir + str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
#     exit()


reg_norm_lst = ['CEU']

# reg_norm_lst = ['ANT', 'RUA', 'SJM', 'CAU', 'GRL', 'ISL', 'NZL', 'TRP', 'CEU', 'WNA']

### 4. ADD NORMALIZED SERIES FROM NEIGHBOURING GLACIERS TO EXTEND ANOMALIES BACK IN TIME

for region in reg_norm_lst:
    # region = 'ISL'
    spt_anom_fill_lst = []
    spt_anom_sig_fill_lst = []
    print('working on region, ', region)

    spt_anom_in = out_reg_dir + str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'
    spt_anom_df = pd.read_csv(spt_anom_in, delimiter=',', header=0, index_col=0)

    sig_spt_anom_in = out_reg_dir + str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'
    sig_spt_anom_df = pd.read_csv(sig_spt_anom_in, delimiter=',', header=0, index_col=0)

    fog_id_lst = spt_anom_df.columns.to_list()

    for fog_id in fog_id_lst:
        print('working on id, ', fog_id)
        # fog_id='23697'
        max_sig = sig_spt_anom_df[fog_id].max().max()
        print(max_sig)

        STD_id = spt_anom_df[fog_id].loc[spt_anom_df[fog_id].index.isin(list(reference_period))].std()
        print('std: ', STD_id)

        if region == 'ISL': ## Get series from Storbreen, Aalfotbreen and Rembesdalskaaka to normalize (SCA, fog_ids 302, 317, 2296)
            neighbour_anom_in = out_anom_dir + 'SCA_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + 'SCA_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

            # norm_all_neighbour_fog_id = norm_neighbour * STD_id
            # ## observe correlation matrix between regional glacier anomalies and added neighbours
            # anom_in = out_anom_dir + 'ISL_all_reg_gla_anomalies.csv'
            # anom_df = pd.read_csv(anom_in, delimiter=',', header=0, usecols= ['YEAR', '3089'], index_col=['YEAR'])
            # corr = pd.concat([norm_all_neighbour_fog_id, anom_df], axis=1).dropna()
            # print(corr)
            # corr_df=corr.corr(method='pearson')
            # print(corr_df.round(decimals=2))
            # exit()

        if region in ['SJM', 'RUA']: ## Get series from Storglacieren to normalize (SCA, fog_ids 332)
            neighbour_anom_in = out_anom_dir + 'SCA_all_reg_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','332'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + 'SCA_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '332'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

            # ## observe correlation matrix between regional glacier anomalies and added neighbours
            # anom_in = out_anom_dir + 'SJM_all_reg_gla_anomalies.csv'
            # anom_df = pd.read_csv(anom_in, delimiter=',', header=0, usecols= ['YEAR', '291', '292'], index_col=['YEAR'])
            # corr = pd.concat([neighbour_anom_df, anom_df], axis=1).dropna()
            # print(corr)
            # corr_df=corr.corr(method='pearson')
            # print(corr_df.round(decimals=2))
            # exit()

        if region == 'CEU': ## Get series from Claridenfirn (CEU, fog_ids 2660)
            neighbour_anom_in = out_anom_dir + 'CEU_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','2660' ], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + 'CEU_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '4617', '4619', '4620'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region == 'WNA': ## Get series from Taku glacier (ALA, fog_ids 124)
            neighbour_anom_in = out_anom_dir + 'WNA_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','124' ], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + 'ALA_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '124'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region == 'CAU': ## Get series from Hinteeisferner, Kesselwand (CEU, fog_ids 491,507)
            neighbour_anom_in = out_anom_dir + 'CEU_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','491', '507' ], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + 'CEU_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '491', '507'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region == 'GRL': ## Get series from Meighen and Devon Ice Caps to normalize (ACN, fog_ids 16, 39)
            neighbour_anom_in = out_anom_dir + 'GRL_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR', '16', '39'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + 'ACN_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '102349', '104095'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

            # ## observe correlation matrix between regional glacier anomalies and added neighbours
            # anom_in = out_anom_dir + 'GRL_all_reg_gla_anomalies.csv'
            # anom_df = pd.read_csv(anom_in, delimiter=',', header=0, usecols= ['YEAR', '3350', '1629'], index_col=['YEAR'])
            # anom_in2 = out_anom_dir + 'SCA_all_reg_gla_anomalies.csv'
            # anom_df2 = pd.read_csv(anom_in2, delimiter=',', header=0, usecols= ['YEAR', '332', '302'], index_col=['YEAR'])
            # df=pd.concat([anom_df, anom_df2], axis=1).dropna()
            # corr = pd.concat([neighbour_anom_df, df], axis=1).dropna()
            # print(corr)
            # corr_df=corr.corr(method='pearson')
            # print(corr_df.round(decimals=2))
            # exit()

        if region in ['ANT', 'NZL', 'SA1', 'TRP']: ## Get series from Echaurren to normalize (SA2, fog_id 1344)
            neighbour_anom_in = out_anom_dir + 'SA2_all_reg_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','1344'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + 'SA2_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '1344'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / (STD_neigbour)
            print('std: ', STD_neigbour)

        norm_all_neighbour_fog_id = norm_neighbour * STD_id

        norm_neighbour_fog_id = norm_all_neighbour_fog_id.mean(axis=1)
        norm_neighbour_fog_id = pd.DataFrame(norm_neighbour_fog_id, columns=[str(fog_id)])
        fog_id_spt_anom = spt_anom_df.filter([fog_id], axis=1)

        id_anom_fill = fog_id_spt_anom.fillna(norm_neighbour_fog_id)
        spt_anom_fill_lst.append(id_anom_fill)

        # fill past uncertainties
        id_sig_past_df = np.sqrt(max_neighbour_sig_mean_anom.pow(2) + max_sig ** 2)
        # print(id_sig_past_df)
        # exit()
        # print(sig_spt_anom_df[fog_id])

        sig_spt_anom_df[fog_id] = sig_spt_anom_df[fog_id].fillna(id_sig_past_df)
        # sig_spt_anom_df[fog_id].to_csv(out_long_dir+'test.csv')
        # exit()
        spt_anom_sig_fill_lst.append(sig_spt_anom_df[fog_id])

        # exit()

    reg_anom_fill_df = pd.concat(spt_anom_fill_lst, axis='columns')
    reg_anom_fill_df.to_csv(out_long_dir + str(region) + '_spt_anoms_fill_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')

    reg_anom_sig_fill_df = pd.concat(spt_anom_sig_fill_lst, axis='columns')
    reg_anom_sig_fill_df.to_csv(out_long_dir + str(region) + '_spt_ERRORs_fill_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
    # exit()



print('.........................................................................................')
print('"The End"')
print('.........................................................................................')
exit()
