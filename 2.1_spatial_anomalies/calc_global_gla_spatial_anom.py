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

out_dir = os.path.join(path, 'out_data_'+fog_version+'_review')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# create directory for regional glaciers anomalies
out_reg_dir= os.path.join(out_dir, 'MEAN_spatial_gla_anom_ref_'+str(year_ini)+'-'+str(year_fin))
if not os.path.exists(out_reg_dir):
    os.mkdir(out_reg_dir)

out_anom_dir= os.path.join(out_dir, 'LOOKUP_spatial_and_reg_ids_ref_'+str(year_ini)+'-'+str(year_fin))
if not os.path.exists(out_anom_dir):
    os.mkdir(out_anom_dir)

out_long_dir= os.path.join(out_dir, 'LONG-NORM_spatial_gla_anom_ref_' + str(year_ini) + '-' + str(year_fin))
if not os.path.exists(out_long_dir):
    os.mkdir(out_long_dir)

##### 2.1 READ MASS BALANCE DATA ######

# read FoG file with global annual and seasonal mass-balance data
in_data_gla = os.path.join(path, 'in_data', 'fog-'+fog_version,'fog_bw-bs-ba_'+fog_version+'.csv')
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

ba_file = os.path.join(path, 'in_data', 'fog-'+fog_version, 'fog_' + fog_version+ '_ba.csv')
ba_unc_file = os.path.join(path, 'in_data', 'fog-'+fog_version, 'fog_' + fog_version+ '_ba_unc.csv')

# ba_df = create_mb_dataframe(input_gla_df, all_fog_gla_id_lst, yr_lst, 'ANNUAL_BALANCE')
# ba_df.to_csv(ba_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')
# ba_unc_df = create_mb_dataframe(input_gla_df, all_fog_gla_id_lst, yr_lst, 'ANNUAL_BALANCE_UNC')
# ba_unc_df.to_csv(ba_unc_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')
# exit()

# read FoG file with global annual mass-balance data
ba_df = pd.read_csv(ba_file, delimiter=',', header=0, index_col=0)
ba_df.columns = ba_df.columns.map(int)  # make columns names great again

### Add missing years to Urumqi glacier fog_id 853
file= os.path.join(path, 'in_data', 'urumqi_missing_years.csv')
df = pd.read_csv(file, delimiter=',', header=0, index_col=0)
df.columns = df.columns.map(int)  # make columns names great again
ba_df = ba_df.fillna(df)

ba_unc_df = pd.read_csv(ba_unc_file, delimiter=',', header=0, index_col=0)
ba_unc_df.columns = ba_unc_df.columns.map(int)  # make columns names great again

in_gla_coord = os.path.join(path, 'in_data','fog-'+fog_version, 'FOG_coord_'+fog_version+'.csv')
coord_gla_df= pd.read_csv(in_gla_coord, encoding='latin1', delimiter=',', header=0, index_col='WGMS_ID').sort_index()

##### 2.2 READ GEODETIC DATA ######

# read FoG file with global geodetic data
in_data_geo = os.path.join(path, 'in_data', 'fog-'+fog_version, '_FOG_GEO_MASS_BALANCE_DATA_'+fog_version+'.csv')
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
# reg_lst = ['CAU']

for region in reg_lst:
    # region= 'CAU'
    print('Working on region, ', region)

    ## create empty dataframes for spatial anomalies and uncertainties
    spt_anom_df = pd.DataFrame(index=yr_lst)
    spt_anom_df.index.name = 'YEAR'
    spt_anom_lst = []

    spt_anom_err_df = pd.DataFrame(index=yr_lst)
    spt_anom_err_df.index.name = 'YEAR'
    sig_spt_anom_lst = []

    ## number crunching: SELECT GEODETIC DATA FOR GLACIER REGION GROUP

    if region == 'SA1':
        reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-01')]
    elif region == 'SA2':
        reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-02')]
    else:
        reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_REGION_CODE'] == str(region))]

    ## create a list of fog_ids with geodetic data for the region group
    reg_fog_geo_id_lst = reg_geo_df['WGMS_ID'].unique().tolist()

    ############################################################################################################################
    ###### 3. CALCULATING SPATIAL ANOMALIES ######

    ## SELECT MASS BALANCE DATA FOR GLACIER REGION GROUP
    ## create list of glacier mass balance series ids possible to calculate the glacier temporal variabiity or anomaly
    ## remove or add neighbouring glacier mass balance series

    if region == 'ASN': # add Urumqui, remove Hamagury yuki, add
        add_id_lst = [853, 817]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
        rem_id = 897  # Hamagury yuki (ASN)
        rem_id_lst2 = [897, 1511, 1512]  # Hamagury yuki (ASN), Urumqi East and west branches (ASC)
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ALA')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASC')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst2)].index)
        glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'] == rem_id].index)
        # print(list(glac['WGMS_ID'].unique().tolist()))
        # exit()

    if region == 'ASE':
        add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
        rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASC')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
        glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]

    if region == 'ASC':
        rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASE')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
        glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'].isin(rem_id_lst)].index)

    if region == 'ASW':
        add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
        rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASC')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASE')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
        glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]

    if region == 'CEU':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = glac

    if region == 'SA1':
        rem_id_lst = [3902, 3903, 3904, 3905, 1344, 3972]  # keep Martial Este only
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
        glac_reg = glac
        # print(list(glac['WGMS_ID'].unique().tolist()))
        # exit()

    if region == 'SA2':  # keep Echaurren Norte only
        rem_id_lst = [3902, 3903, 3904, 3905, 2000, 3972]
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
        glac_reg = glac

    if region == 'NZL':
        add_id_lst = [2000]  # Martial Este (SAN-01)
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = glac

    if region == 'ANT':
        rem_id_lst = [878, 3973]  # Dry valley glaciers
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
        glac_reg = glac

    if region == 'RUA':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'SJM') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = glac

    if region == 'SJM':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = glac

    if region == 'ALA':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'WNA') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

    if region == 'WNA':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ALA'), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

    if region == 'TRP':
        rem_id = 226  # Yanamarey
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac = glac.drop(glac[glac['WGMS_ID'] == rem_id].index)
        glac_reg = glac

    if region == 'ACS':
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = glac

    if region == 'ACN':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = glac

    if region == 'GRL':
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

    if region == 'ISL':
        glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'GRL')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

    if region == 'SCA':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = glac

    if region == 'CAU':
        glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
        glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

    ## Find all possible individual glacier anomalies (with respect to reference period) for the given glacier id

    ## number crunching:   select mass-balance data for glacier region groups
    ba_glac_df = ba_df.loc[:, list(glac['WGMS_ID'].unique().tolist())]
    glac_anom = calc_anomalies(ba_glac_df, reference_period, region)
    unc_glac_anom = calc_spt_anomalies_unc(glac_anom, ba_unc_df, glac_anom.columns.to_list())

    # FOR SA2 ONLY: if no uncertainty measurement use the regional annual mean uncertainty of the glaciological sample
    if unc_glac_anom.isnull().sum().sum():
        for id in unc_glac_anom.columns.tolist():
            year_min = glac_anom[id].first_valid_index()
            yrs = list(range(1915, year_min))
            unc_glac_anom[id].fillna(np.nanmean(ba_unc_df), inplace=True)
            unc_glac_anom[id].mask(unc_glac_anom.index.isin(yrs), np.nan, inplace=True)
    else:
        continue

    ## Correct suspicious anomaly from Echaurren Norte by normalizing past period to present period amplitude.
    if region == 'SA2':
        STD_ech_ok = glac_anom.loc[glac_anom.index.isin(list(range(2004, (2022 + 1))))].std()
        STD_ech_bad = glac_anom.loc[glac_anom.index.isin(list(range(1980, (1999 + 1))))].std()
        glac_anom_pres_ok = glac_anom.loc[glac_anom.index >= 2004]
        norm_past = glac_anom.loc[glac_anom.index.isin(list(range(1885, (2003 + 1))))] / STD_ech_bad
        glac_anom_past_new = (norm_past * STD_ech_ok).round(decimals=1)
        glac_anom = pd.concat([glac_anom_past_new, glac_anom_pres_ok], axis = 0)

    # ## Filter series for regional anomaly to use
    ba_reg_glac_df = ba_df.loc[:, list(glac_reg['WGMS_ID'].unique().tolist())]
    reg_glac_anom = calc_anomalies(ba_reg_glac_df, reference_period, region)

    # ## Correct suspicious anomaly from Echaurren Norte by normalizing past period to present period amplitude.
    if region == 'SA2':
        STD_ech_ok = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(2004, (2022 + 1))))].std()
        STD_ech_bad = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1980, (1999 + 1))))].std()
        reg_glac_anom_pres_ok = reg_glac_anom.loc[reg_glac_anom.index >= 2004]
        norm_past = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1885, (2003 + 1))))] / STD_ech_bad
        reg_glac_anom_past_new = (norm_past * STD_ech_ok).round(decimals=1)
        reg_glac_anom = pd.concat([glac_anom_past_new, glac_anom_pres_ok], axis = 0)

    # ## select close anomalies for calculating the fog_id glacier anomaly
    spatial_id_fin_lst = glac_anom.columns.to_list()
    # print(spatial_id_fin_lst)

    close_gla_weights = coord_gla_df.loc[spatial_id_fin_lst, :]
    lat_glac = close_gla_weights['LATITUDE']
    lon_glac= close_gla_weights['LONGITUDE']

    # ROMAIN: Replacing by inverse-distance weighting by kriging here
    anoms_4_fog_id_df = glac_anom[spatial_id_fin_lst]
    unc_anoms_4_fog_id_df = unc_glac_anom[spatial_id_fin_lst]

    # Get variance of anomalies in this region for the kriging algorithm
    var_anom = np.nanvar(anoms_4_fog_id_df)

    # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
    # to compute the kriging

    from kriging import wrapper_latlon_krige_ba_anom
    arr_mean_anom = np.ones((len(anoms_4_fog_id_df.index), len(reg_fog_geo_id_lst)), dtype=np.float32)
    arr_sig_anom = np.ones((len(anoms_4_fog_id_df.index), len(reg_fog_geo_id_lst)), dtype=np.float32)
    for i in range(len(anoms_4_fog_id_df.index)):
        print(f"Kriging region {region} for year {anoms_4_fog_id_df.index[i]}")

        # Create dataframe with anomalies, lat and lon
        yearly_anom_df = anoms_4_fog_id_df.iloc[i, :]

        obs_df = pd.DataFrame(data={"ba_anom": yearly_anom_df.values, "lat": np.array(lat_glac), "lon": np.array(lon_glac)})

        # print(obs_df)
        valids = np.isfinite(obs_df["ba_anom"])

        # If no data is valid, write NaNs
        if np.count_nonzero(valids) < 1:
            arr_mean_anom[i, :] = np.nan
            arr_sig_anom[i, :] = np.nan
            continue
        # Otherwise limit to valid data only
        else:
            obs_df = obs_df[valids]

        # Get latitude and longitude of unobserved glacier to predict
        lat_id = coord_gla_df.loc[reg_fog_geo_id_lst, 'LATITUDE']
        lon_id = coord_gla_df.loc[reg_fog_geo_id_lst, 'LONGITUDE']

        # Create dataframe with points where to predict (could be several at once but here always one)
        pred_df = pd.DataFrame(data={"lat": lat_id, "lon": lon_id})

        # Kriging at the coordinate of the current glacier
        mean_anom, sig_anom = wrapper_latlon_krige_ba_anom(df_obs=obs_df, df_pred=pred_df, var_anom=var_anom)
        arr_mean_anom[i, :] = mean_anom
        arr_sig_anom[i, :] = sig_anom
        # print(mean_anom)
        # print(arr_mean_anom)
        # exit()

    # And write back the 1D list of uncertainties into an indexed (by YEAR) dataframe
    anom_fog_id_df = pd.DataFrame(index=anoms_4_fog_id_df.index, data=arr_mean_anom, columns=[str(fog_id) for fog_id in reg_fog_geo_id_lst])
    sig_anom_df = pd.DataFrame(index=anoms_4_fog_id_df.index, data=arr_sig_anom, columns=[str(fog_id) for fog_id in reg_fog_geo_id_lst])

    ## CALCULATE:  mean anomaly for fog_id
    ## if glacier has in situ measurements i.e. dist = 0 use the own glaciers anomaly
    anom_fog_id_df = anom_fog_id_df.loc[anom_fog_id_df.index >= 1915]
    spt_anom_lst.append(anom_fog_id_df)

    ## CALCULATE: Uncertainty for fog_id
    sig_anom_df = round(sig_anom_df, 2)
    sig_anom_df = sig_anom_df.loc[sig_anom_df.index >= 1915]
    sig_spt_anom_lst.append(sig_anom_df)
    # print(sig_anom_df)
    # exit()

    # ###### PLOT glacier anomalies and Regional mean
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # out_plot_path = os.path.join(path, 'out_data_'+fog_version+'_review','plot_reg_anomaly_ref_period_' + str(year_ini) + '-' + str(year_fin) + '_final')
    # if not os.path.exists(out_plot_path):
    #     os.mkdir(out_plot_path)
    # # print(anoms_4_fog_id_df)
    # # exit()
    # anoms_4_fog_id_df = anoms_4_fog_id_df/1000
    #
    # if len(anoms_4_fog_id_df.columns) > 1:
    #     plot_anom_df = anoms_4_fog_id_df.mul(1)
    #     plot_anom_df['UNC'] =  sig_anom_df/1000
    #     plot_anom_df['MEAN'] = anom_fog_id_df/1000
    #
    #     ax = anoms_4_fog_id_df.plot(color='grey', linewidth=0.5, legend=False)
    #     ax.set_ylim([-3, 3])
    #     ax.axhline(0, color='Grey', linewidth=1)
    #     ax.set_ylabel('\u03B2 (m w.e.)', size=18, weight=600)
    #     ax.set_xlabel('Year', size=18, weight=600)
    #     ax.text(1995, 3, 'N = ' + str(len(anoms_4_fog_id_df.columns)) + ' glaciers', size=14, weight=600)
    #
    #     plot_anom_df['MEAN'].plot(ax=ax, color='black', alpha=0.7, linewidth=2)
    #     plt.fill_between(plot_anom_df.index, plot_anom_df['MEAN'] + plot_anom_df['UNC'],
    #                      plot_anom_df['MEAN'] - plot_anom_df['UNC'], color='grey', alpha=0.3, linewidth=0)
    #
    #     # save plot
    #     ax.tick_params(labelsize=18)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.set_xlim([1950, 2023])
    #     plt.xticks(np.arange(1960, 2023, 20))
    #
    #     out_fig = os.path.join(out_plot_path, 'Anomaly_and_UNC_for_id_' + str(fog_id) + '_ref_period_' + str(year_ini) + '-' + str(
    #         year_fin) + '_' + fog_version + '.svg')
    #
    #     fig.tight_layout()
    #     plt.savefig(out_fig, dpi=300)
    #     print('Plot saved as {}.'.format(out_fig))
    #
    #     plt.close()
    #     exit()
    # else:
    #     print('................... Region with only one glacier anomaly:', region,
    #           '............................')

    glac_anom.to_csv(os.path.join(out_anom_dir, region + '_all_SEL_gla_anomalies.csv'))
    reg_glac_anom.to_csv(os.path.join(out_anom_dir, region + '_all_reg_gla_anomalies.csv'))
    unc_glac_anom.to_csv(os.path.join(out_anom_dir, region + '_all_SEL_gla_anomalies_UNC.csv'))

    ### Save all glacier anomalies and uncertainties - exclude uncertainties from the SAN regions
    spt_anom_df = pd.concat(spt_anom_lst, axis='columns')
    spt_anom_df.to_csv(os.path.join(out_reg_dir, str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))

    sig_spt_anom_df = pd.concat(sig_spt_anom_lst, axis='columns')
    sig_spt_anom_df.to_csv(os.path.join(out_reg_dir, str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))

    print("--- %s seconds ---" % (time.time() - read_time))

    ### Save glacier anomalies and uncertainties OK with long time periods
    reg_ok_lst = ['ACS', 'ACN', 'ASW', 'ASE', 'ASC', 'ASN', 'ALA', 'SCA', 'SA2']
    if region in reg_ok_lst:
        spt_anom_df.to_csv(os.path.join(out_long_dir, str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))
        sig_spt_anom_df.to_csv(os.path.join(out_long_dir, str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))



reg_norm_lst = ['ANT', 'RUA', 'SJM', 'CAU', 'GRL', 'SA1', 'ISL', 'NZL', 'TRP', 'CEU', 'WNA']

### 4. ADD NORMALIZED SERIES FROM NEIGHBOURING GLACIERS TO EXTEND ANOMALIES BACK IN TIME

for region in reg_norm_lst:
    # region = 'SA1'
    spt_anom_fill_lst = []
    spt_anom_sig_fill_lst = []
    print('working on region, ', region)

    spt_anom_in = os.path.join(out_reg_dir, str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
    spt_anom_df = pd.read_csv(spt_anom_in, delimiter=',', header=0, index_col=0)

    sig_spt_anom_in = os.path.join(out_reg_dir, str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
    sig_spt_anom_df = pd.read_csv(sig_spt_anom_in, delimiter=',', header=0, index_col=0)

    fog_id_lst = spt_anom_df.columns.to_list()

    for fog_id in fog_id_lst:
        print('working on id, ', fog_id)
        # fog_id='23697'
        max_sig = sig_spt_anom_df[fog_id].max().max()

        STD_id = spt_anom_df[fog_id].loc[spt_anom_df[fog_id].index.isin(list(reference_period))].std()
        print('std: ', STD_id)

        if region == 'ISL': ## Get series from Storbreen, Aalfotbreen and Rembesdalskaaka to normalize (SCA, fog_ids 302, 317, 2296)
            neighbour_anom_in = out_anom_dir + '\\SCA_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + '\\SCA_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region in ['SJM', 'RUA']: ## Get series from Storglacieren to normalize (SCA, fog_ids 332)
            neighbour_anom_in = out_anom_dir + '\\SCA_all_reg_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','332'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + '\\SCA_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '332'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region == 'CEU':  ## Get series from Claridenfirn (CEU, fog_ids 2660)
            neighbour_anom_in = out_anom_dir + '\\CEU_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '2660'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + '\\CEU_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols=['YEAR', '4617', '4619', '4620'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region == 'WNA':  ## Get series from Taku glacier (ALA, fog_ids 124)
            neighbour_anom_in = out_anom_dir + '\\WNA_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '124'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + '\\ALA_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols=['YEAR', '124'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region == 'CAU':  ## Get series from Hinteeisferner, Kesselwand (CEU, fog_ids 491,507)
            neighbour_anom_in = out_anom_dir + '\\CEU_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '491', '507'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + '\\CEU_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0,usecols=['YEAR', '491', '507'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region == 'GRL':  ## Get series from Meighen and Devon Ice Caps to normalize (ACN, fog_ids 16, 39)
            neighbour_anom_in = out_anom_dir + '\\GRL_all_SEL_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '16', '39'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + '\\ACN_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols=['YEAR', '102349', '104095'], index_col=['YEAR'])
            max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
            STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
            norm_neighbour = neighbour_anom_df / STD_neigbour
            print('std: ', STD_neigbour)

        if region in ['ANT', 'NZL', 'SA1', 'TRP']: ## Get series from Echaurren to normalize (SA2, fog_id 1344)
            neighbour_anom_in = out_anom_dir + '\\SA2_all_reg_gla_anomalies.csv'
            neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','1344'], index_col=['YEAR'])
            neighbour_sig_mean_anom_in = out_reg_dir + '\\SA2_spt_ERRORs_ref_2011-2020_2024-01.csv'
            neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, index_col=['YEAR'])
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
        sig_spt_anom_df[fog_id] = sig_spt_anom_df[fog_id].fillna(id_sig_past_df)
        spt_anom_sig_fill_lst.append(sig_spt_anom_df[fog_id])

        # exit()

    reg_anom_fill_df = pd.concat(spt_anom_fill_lst, axis='columns')
    reg_anom_fill_df.to_csv(os.path.join(out_long_dir, str(region) + '_spt_anoms_fill_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))

    reg_anom_sig_fill_df = pd.concat(spt_anom_sig_fill_lst, axis='columns')
    reg_anom_sig_fill_df.to_csv(os.path.join(out_long_dir, str(region) + '_spt_ERRORs_fill_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))




print('.........................................................................................')
print('"The End"')
print('.........................................................................................')
exit()
