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
import seaborn as sns
# import geopandas as gpd
from matplotlib.ticker import PercentFormatter
from functions_ggmc import *
# from propagation_functions import wrapper_latlon_double_sum_covar, sig_dh_spatialcorr, sig_rho_dv_spatialcorr, ba_anom_spatialcorr
from scipy.stats import norm
pd.options.mode.chained_assignment = None  # default='warn'
import scipy as sp  # usage: scipy.stats.funct()

# reg_lst= ['ASE', 'ASW', 'ASC']
# reg_lst= [ 'ISL', 'CEU', 'ALA', 'ASW', 'SA2','WNA', 'ACN', 'ACS', 'GRL', 'SJM', 'SCA', 'RUA', 'ASN', 'CAU', 'ASC', 'ASE', 'TRP', 'SA1', 'NZL', 'ANT']
# reg_lst= [ 'ISL', 'CEU', 'ALA', 'ASW', 'SA2','WNA', 'ACN', 'ACS', 'GRL', 'SJM', 'SCA', 'RUA', 'ASN', 'CAU', 'ASC', 'ASE', 'TRP', 'SA1', 'NZL', 'ANT']
reg_lst= [ 'ISL', 'CEU', ]

rgi_region= {'ACN' : 'Arctic Canada (North)', 'WNA' : 'Western Canada & US', 'ALA' : 'Alaska', 'ACS' : 'Arctic Canada (South)', 'TRP' : 'Low Latitudes', 'SCA' : 'Scandinavia',
             'SJM' : 'Svalbard', 'CEU' : 'Central Europe', 'CAU' : 'Caucasus & Middle East', 'ASC' : 'Central Asia', 'ASN' : 'North Asia', 'ASE' : 'South Asia (East)',
             'NZL' : 'New Zealand', 'ASW' : 'South Asia (West)', 'GRL' : 'Greenland Periphery', 'ANT' : 'Antarctic & Subantarctic', 'ISL' : 'Iceland', 'RUA' : 'Russian Arctic',
             'SA1' : 'Southern Andes (Patagonia)', 'SA2' : 'Southern Andes (Central)'}

rgi_reg= {'ACN' : 'ArcticCanadaNorth', 'WNA' : 'WesternCanadaUS', 'ALA' : 'Alaska', 'ACS' : 'ArcticCanadaSouth', 'TRP' : 'LowLatitudes', 'SCA' : 'Scandinavia',
             'SJM' : 'Svalbard', 'CEU' : 'CentralEurope', 'CAU' : 'CaucasusMiddleEast', 'ASC' : 'CentralAsia', 'ASN' : 'NorthAsia', 'ASE' : 'SouthAsiaEast',
             'NZL' : 'NewZealand', 'ASW' : 'SouthAsiaWest', 'GRL' : 'GreenlandPeriphery', 'ANT' : 'AntarcticSubantarctic', 'ISL' : 'Iceland', 'RUA' : 'RussianArctic',
             'SAN' : 'SouthernAndes', 'SA1' : 'SouthernAndes', 'SA2' : 'SouthernAndes'}

rgi_code= {'ALA' : '01', 'WNA' : '02', 'ACN' : '03', 'ACS' : '04', 'GRL' : '05', 'ISL' : '06', 'SJM' : '07', 'SCA' : '08', 'RUA' : '09', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SA1' : '17', 'SA2' : '17', 'NZL' : '18', 'ANT' : '19'}

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
path_proj = os.path.abspath(os.path.join(__file__ ,"../.."))

fog_version = '2024-01'

path_oce = os.path.join(path_proj, '3.2_global_CE_spatial_anomaly_RH', 'out_data_'+fog_version+'_review','OCE_files_by_region') # path to regional OCE files

#################################################################################################
##    Define parameters
#################################################################################################
out_dir = os.path.join(path, 'out_data_'+fog_version+'_review')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Define correlation factor for error propagation
# corr_factor = 250 # from Zemp et al. 2019

DM_series_min_yr = 1915 # starting year of DM series saved in the .csv files

# end_plot= '_OCE_and_B_reg_' # plots Regional Mass balance and the individual region OCE
end_plot = '_B_reg_and_err_' # plots Regional Mass balance and the relative mass balance uncertainty
# end_plot = '_B_reg_extremes'
# end_plot ='_stats_B_anomalies_reg_'
# end_plot ='_stats_norm_std_reg_'
# end_plot =''

plt_year_min = 1915 # starting year for Regional series plot
plt_year_max = 2025 # end year for Regional series plot

axes = 'eq_axes' # all region with same Y axes, visualizes best the contribution between regions
# axes = 'tight' # fits Y axes to each region, visualizes best the temporal variability of each region

#################################################################################################
##    READ ID links and areas
#################################################################################################
# in_data_coords = os.path.join(path, 'in_data', 'fog-'+fog_version, '_FOG_GEO_MASS_BALANCE_DATA_'+fog_version+'.csv')
# data_coords_df = pd.read_csv(in_data_coords, encoding='utf-8', delimiter=',', header=0, usecols=['WGMS_ID','LATITUDE', "LONGITUDE"])
# data_coords_df = data_coords_df.groupby('WGMS_ID').nth(0)

in_data_rgi_area = os.path.join(path, 'in_data', '_RGI_All_ID_Area.csv')
id_rgi_area_df = pd.read_csv(in_data_rgi_area, encoding='latin1', delimiter=',', header=0)

in_data_glims = os.path.join(path, 'in_data', 'CAU_glims_attribute_table.csv')
id_glims_coords_df = pd.read_csv(in_data_glims, encoding='latin1', delimiter=',', header=0 ,usecols= ['glac_id','db_area','CenLat', 'CenLon'])
id_glims_coords_df = id_glims_coords_df.rename(columns={'glac_id': 'RGIId'}).set_index('RGIId')

rgi_path= os.path.join(path , 'in_data', '00_rgi60', '00_rgi60_attribs')

in_data_zemp = os.path.join(path, 'in_data', 'zemp_etal_regional_series')

############################################################################################################################

###### Calculate specific glacier mass balance by region ######

Reg_mb_df = pd.DataFrame()
Reg_sig_mb_df = pd.DataFrame()
Reg_sumary_df = pd.DataFrame()

for region in reg_lst:
    # region='ISL'
    print('working on region, ', region)

    ## Define and read input:   regional OCE series and three sources of uncertainty
    in_oce_file = os.path.join(path_oce, region + '_regional_CEs.csv')
    in_sig_dh_oce_file = os.path.join(path_oce, region + '_regional_sigma_dh_CEs.csv')
    in_sig_rho_oce_file = os.path.join(path_oce, region + '_regional_sigma_rho_CEs.csv')
    in_sig_anom_oce_file = os.path.join(path_oce, region + '_regional_sigma_anom_CEs.csv')

    oce_df = pd.read_csv(in_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
    sig_anom_oce_df = pd.read_csv(in_sig_anom_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
    yr = sig_anom_oce_df.first_valid_index()
    # yr=2010
    sig_anom_oce_df = sig_anom_oce_df.loc[sig_anom_oce_df.index >= yr]
    sig_dh_oce_df = pd.read_csv(in_sig_dh_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
    sig_dh_oce_df = sig_dh_oce_df.loc[sig_dh_oce_df.index >= yr]
    sig_rho_oce_df = pd.read_csv(in_sig_rho_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
    sig_rho_oce_df = sig_rho_oce_df.loc[sig_rho_oce_df.index >= yr]

    sig_oce_df = sig_dh_oce_df.copy()
    sig_oce_df = np.sqrt(sig_dh_oce_df**2 + sig_rho_oce_df**2 + sig_anom_oce_df**2)

    nan_lst = sig_oce_df.columns[sig_oce_df.isna().any()].tolist()
    sig_oce_df = sig_oce_df.drop(columns = nan_lst)
    oce_df = oce_df.drop(columns = nan_lst)
    sig_rho_oce_df = sig_rho_oce_df.drop(columns = nan_lst)
    sig_anom_oce_df = sig_anom_oce_df.drop(columns = nan_lst)
    sig_dh_oce_df = sig_dh_oce_df.drop(columns = nan_lst)
    # #
    # print(oce_df)
    # print(sig_oce_df)
    #
    # print(sig_rho_oce_df )
    # print(sig_anom_oce_df )
    # print(sig_dh_oce_df )
    # exit()

    # sig_oce_mean= sig_oce_df.mean().mean()
    #
    # for id in nan_lst:
    #     sig_oce_df[id] = sig_oce_mean

    filename = os.path.join(rgi_path , rgi_code[region]+'_rgi60_'+rgi_reg[region]+'.csv')
    if region == 'GRL':
        rgi_df_all = pd.read_csv(filename, encoding='latin1', delimiter=',', header=0,usecols=['RGIId', 'CenLat', 'CenLon', 'Connect'], index_col=[0])
        rgi_df = rgi_df_all.loc[rgi_df_all['Connect'] != 2]
        l1l2_lst = rgi_df.index.to_list()

    else:
        rgi_df = pd.read_csv(filename, encoding='latin1', delimiter=',', header=0, usecols= ['RGIId', 'CenLat', 'CenLon'], index_col=[0])

    # Keep only glaciers in the region
    if region == 'SA1':
        rgi_area_df= id_rgi_area_df.loc[id_rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-01']
    elif region == 'SA2':
        rgi_area_df= id_rgi_area_df.loc[id_rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-02']
    elif region == 'GRL':
        rgi_area_df = id_rgi_area_df.loc[id_rgi_area_df['RGIId'].isin(l1l2_lst)]
    else:
        rgi_area_df = id_rgi_area_df.loc[(id_rgi_area_df['GLACIER_REGION_CODE'] == region)]

    nb_gla_reg = len(rgi_area_df.index)
    tot_area_rgi_reg = rgi_area_df['AREA'].sum()

    ## select wgms_ids belonging to the region group
    wgms_id_lst = oce_df.columns.to_list()
    wgms_id_lst = [int(i) for i in wgms_id_lst]

    ## Calculate total area of observed glaciers presenting an area value in FoG

    ## Remove glacier IDS with no Area, only for FoG areas
    rgi_area_df = rgi_area_df.set_index('WGMS_ID')
    id_lst=[]
    for id in wgms_id_lst:
        if id in rgi_area_df.index:
            id_lst.append(id)
        else:
            pass

    gla_obs_df = rgi_area_df.loc[id_lst]
    tot_area_obs = gla_obs_df['AREA'].sum()
    nb_gla_obs = len(gla_obs_df)

    gla_obs_df = gla_obs_df.reset_index().set_index('RGIId')

    if region == 'CAU':
        tot_area_rgi_reg = id_glims_coords_df['db_area'].sum()
        gla_obs_area_coord_df = pd.merge(gla_obs_df, id_glims_coords_df, left_index=True, right_index=True).drop_duplicates()
    else:
        gla_obs_area_coord_df = pd.merge(gla_obs_df, rgi_df, left_index=True, right_index=True)

    gla_obs_area_coord_df =gla_obs_area_coord_df.reset_index().set_index('WGMS_ID')
    gla_obs_area_coord_df = gla_obs_area_coord_df[~gla_obs_area_coord_df.index.duplicated()]

    print('total area region / tot nb glaciers in region :  ', tot_area_rgi_reg, ' / ', nb_gla_reg)
    print('total area glaciers observed / number glaciers with observations :  ', tot_area_obs, ' / ', nb_gla_obs)
    # print((tot_area_obs/tot_area_rgi_reg) * 100 -100)
    # print(nb_gla_reg - nb_gla_obs)

    ####### Calculate all glaciers time series and uncertainties ##########

    ## 1. Calculate OCE series for unobserved glaciers as the Weigthed mean from the regional glacier sample with observations
    rel_mb_df = pd.DataFrame()
    rel_sig_dh_mb_df = pd.DataFrame()
    rel_sig_rho_mb_df = pd.DataFrame()
    rel_sig_anom_mb_df = pd.DataFrame()
    rel_sig_mb_df = pd.DataFrame()

    list_df = []
    list_areas = []
    list_lat = []
    list_lon = []
    for id in id_lst:
        # print('working on glacier, ', id)

        # Read area, mass balance estimate and three uncertainty sources
        area= gla_obs_area_coord_df.loc[id, 'AREA']
        lon= gla_obs_area_coord_df.loc[id, 'CenLon']
        lat= gla_obs_area_coord_df.loc[id, 'CenLat']
        mb_oce= oce_df[str(id)]
        sig_dh_oce = sig_dh_oce_df[str(id)]
        sig_rho_oce = sig_rho_oce_df[str(id)]
        sig_anom_oce = sig_anom_oce_df[str(id)]

        # Area weighting for all (uncertainties are also combined pairwise with area-weight in exact propagation)
        mb_oce_rel = (mb_oce * area) / tot_area_obs
        sig_dh_rel = (sig_dh_oce * area)/tot_area_obs
        sig_rho_rel = (sig_rho_oce * area)/tot_area_obs
        sig_anom_rel = (sig_anom_oce * area)/tot_area_obs
        # obs_df["id"] = mb_oce_rel
        # obs_df[id] = sig_rel

        # Dataframes per ID
        rel_mb_df[id] = mb_oce_rel
        # The three error sources
        rel_sig_dh_mb_df[id] = sig_dh_rel
        rel_sig_rho_mb_df[id] = sig_rho_rel
        rel_sig_anom_mb_df[id] = sig_anom_rel

        # The total error
        rel_sig_mb_df[id] = np.sqrt(sig_dh_rel**2 + sig_rho_rel**2 + sig_anom_rel**2)

        # Store lat/lon in a dataframe for "observed" glaciers
        list_lat.append(lat)
        list_lon.append(lon)

    # Area-weighted OCE for observed glaciers
    Aw_oce_obs_df = rel_mb_df.sum(axis=1, min_count=1)
    print(Aw_oce_obs_df)


    ## 2. Calculate OCE uncertainties for observed glaciers

    # Weighted mean Sigma OCE of observed glaciers (only to use later for unobserved glaciers)
    Sig_oce_obs_gla = rel_sig_mb_df.sum(axis=1, min_count=1)
    Sig_dh_mb_gla = rel_sig_dh_mb_df.sum(axis=1, min_count=1)
    Sig_rho_mb_gla = rel_sig_rho_mb_df.sum(axis=1, min_count=1)
    Sig_anom_mb_gla = rel_sig_anom_mb_df.sum(axis=1, min_count=1)

    ## 3. Add OCE series and uncertainties for unobserved glaciers

    # Id -9999 for unobserved glaciers, OCE is the area weighthed average of the regional observed series

    out_oce = os.path.join(out_dir, 'spt_CEs_obs-unobs_per_region')
    if not os.path.exists(out_oce):
        os.mkdir(out_oce)

    oce_df['unobs_gla'] = Aw_oce_obs_df
    oce_df.to_csv(os.path.join(out_oce, region +'_CEs_obs-unobs.csv'))

    sig_oce_df['unobs_gla'] = Sig_oce_obs_gla
    sig_oce_df.to_csv(os.path.join(out_oce, region + '_sigma_tot_CEs_obs-unobs.csv'))

    sig_rho_oce_df['unobs_gla'] = Sig_rho_mb_gla
    sig_rho_oce_df.to_csv(os.path.join(out_oce, region + '_sig_rho_CEs_obs-unobs.csv'))
    sig_anom_oce_df['unobs_gla'] = Sig_anom_mb_gla
    sig_anom_oce_df.to_csv(os.path.join(out_oce, region + '_sigma_anom_CEs_obs-unobs.csv'))
    sig_dh_oce_df['unobs_gla'] = Sig_dh_mb_gla
    sig_dh_oce_df.to_csv(os.path.join(out_oce, region + '_sigma_dh_CEs_obs-unobs.csv'))

exit()
#     ####### Calculate Regional specific mass balance time series ##########
#
#     Reg_mb_df[region] = Aw_oce_obs_df
#     nb_unobs_gla = nb_gla_reg - nb_gla_obs
#
#     # # Fully correlated propagation for residual error of anomaly
#     # Aw_sig_anom_obs_df = rel_sig_anom_mb_df.sum(axis=1, min_count=1)
#
#     # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
#     # to compute the pairwise error propagation for dh and density across all glaciers of that year
#     list_sig_dh_yearly = []
#     list_sig_rho_yearly = []
#     list_sig_anom_yearly = []
#
#     for i in range(len(rel_sig_dh_mb_df.index)):
#
#         print(f"Propagating uncertainties from glaciers to region for year {rel_sig_dh_mb_df.index[i]}")
#
#         # Create dataframe with dh errors, lat and lon
#         yearly_dh_df = rel_sig_dh_mb_df.iloc[i, :]
#         yearly_dh_df["errors"] = yearly_dh_df.values.flatten()
#         yearly_dh_df["lat"] = np.array(list_lat)
#         yearly_dh_df["lon"] = np.array(list_lon)
#
#         # Spatial correlations for dh
#         sig_dh_obs = wrapper_latlon_double_sum_covar(yearly_dh_df, spatialcorr_func=sig_dh_spatialcorr)
#
#         # Check propagation works as intended: final estimate is between fully correlated and independent
#         sig_dh_fullcorr = np.sum(yearly_dh_df["errors"])
#         sig_dh_uncorr = np.sqrt(np.sum(yearly_dh_df["errors"]**2))
#         print(f"{sig_dh_uncorr}, {sig_dh_obs}, {sig_dh_fullcorr}")
#         assert sig_dh_uncorr <= sig_dh_obs <= sig_dh_fullcorr
#
#         # Create dataframe with rho errors, lat and lon
#         yearly_rho_df = rel_sig_rho_mb_df.iloc[i, :]
#         yearly_rho_df["errors"] = yearly_rho_df.values
#         yearly_rho_df["lat"] = np.array(list_lat)
#         yearly_rho_df["lon"] = np.array(list_lon)
#
#         # Spatial correlation for rho for a 1-year period
#         def sig_rho_dv_spatialcorr_yearly(d):
#             return sig_rho_dv_spatialcorr(d, dt=1)
#         sig_rho_obs = wrapper_latlon_double_sum_covar(yearly_rho_df, spatialcorr_func=sig_rho_dv_spatialcorr_yearly)
#
#         # Check propagation works as intended: final estimate is between fully correlated and independent
#         sig_rho_fullcorr = np.sum(yearly_rho_df["errors"])
#         sig_rho_uncorr = np.sqrt(np.sum(yearly_rho_df["errors"] ** 2))
#         # print(f"{sig_rho_uncorr}, {sig_rho_obs}, {sig_rho_fullcorr}")
#         assert sig_rho_uncorr <= sig_rho_obs <= sig_rho_fullcorr
#
#         # Create dataframe with anom errors, lat and lon
#         yearly_anom_df = rel_sig_anom_mb_df.iloc[i, :]
#         # print(yearly_anom_df)
#         # exit()
#         yearly_anom_df["errors"] = yearly_anom_df.values
#         yearly_anom_df["lat"] = np.array(list_lat)
#         yearly_anom_df["lon"] = np.array(list_lon)
#
#         # Spatial correlations for anom
#         sig_anom_obs = wrapper_latlon_double_sum_covar(yearly_anom_df, spatialcorr_func=ba_anom_spatialcorr)
#
#         # Check propagation works as intended: final estimate is between fully correlated and independent
#         sig_anom_fullcorr = np.sum(yearly_anom_df["errors"])
#         sig_anom_uncorr = np.sqrt(np.sum(yearly_anom_df["errors"]**2))
#         # print(f"{sig_anom_uncorr}, {sig_anom_obs}, {sig_anom_fullcorr}")
#         assert sig_anom_uncorr <= sig_anom_obs <= sig_anom_fullcorr
#
#         # Append to list for each yearly period
#         list_sig_dh_yearly.append(sig_dh_obs)
#         list_sig_rho_yearly.append(sig_rho_obs)
#         list_sig_anom_yearly.append(sig_anom_obs)
#
#     # And write back the 1D list of uncertainties into an indexed (by YEAR) dataframe
#     Aw_sig_dh_obs_df =  pd.DataFrame(index=sig_anom_oce_df.index.copy())
#     Aw_sig_dh_obs_df['dh']= list_sig_dh_yearly
#
#     Aw_sig_rho_obs_df = pd.DataFrame(index=sig_anom_oce_df.index.copy())
#     Aw_sig_rho_obs_df['rho'] = list_sig_rho_yearly
#
#     Aw_sig_anom_obs_df = pd.DataFrame(index=sig_anom_oce_df.index.copy())
#     Aw_sig_anom_obs_df['anom']= list_sig_anom_yearly
#
#     print(Aw_sig_dh_obs_df)
#     print(Aw_sig_rho_obs_df)
#     print(Aw_sig_anom_obs_df)
#
#     Sig_oce_obs_propag = np.sqrt(Aw_sig_dh_obs_df['dh']**2 + Aw_sig_rho_obs_df['rho']**2 + Aw_sig_anom_obs_df['anom']**2)
#     print(Sig_oce_obs_propag)
#     # exit()
#
#     # Defining area-weighted uncertainty of unobserved glaciers based on the mean uncertainty of observed glaciers
#     area_unobs = round(tot_area_rgi_reg, 2) - round(tot_area_obs, 2)
#     sig_W_unobs = Sig_oce_obs_gla * (area_unobs / tot_area_rgi_reg)
#
#     # Area-weight the observed glaciers before combining in final uncertainty
#     sig_W_obs = Sig_oce_obs_propag * (tot_area_obs / tot_area_rgi_reg)
#
#
#     # Final regional uncertainty!
#     reg_sig = np.sqrt(sig_W_obs**2 + sig_W_unobs**2)
#
#     Reg_sig_mb_df[region] = reg_sig
#
#     Reg_sumary_df['Aw_B m w.e.'] = Aw_oce_obs_df/1000
#     Reg_sumary_df['sigma_B m w.e.'] = reg_sig / 1000
#     Reg_sumary_df['sigma_propagated m w.e.'] = Sig_oce_obs_propag / 1000
#     Reg_sumary_df['sigma_dh m w.e.'] = Aw_sig_dh_obs_df / 1000
#     Reg_sumary_df['sigma_rho m w.e.'] = Aw_sig_rho_obs_df / 1000
#     Reg_sumary_df['sigma_anom m w.e.'] = Aw_sig_anom_obs_df / 1000
#     Reg_sumary_df.to_csv(os.path.join(out_dir, region +'_B_and_sigma.csv'))
#     # exit()
#
#     # reg_sig_asw= pd.read_csv(os.path.join(out_dir, 'ASW_B_and_sigma.csv'), encoding='latin1', delimiter=',', header=0, index_col= 'YEAR')
#     # reg_sig_asc= pd.read_csv(os.path.join(out_dir, 'Regional_B_series_AreaWeighted_edited.csv'), encoding='latin1', delimiter=',', header=0, index_col= 'YEAR')
#     # Aw_oce_obs_df = reg_sig_asc['ASC']
#     # reg_sig = reg_sig_asw['sigma_B m w.e.']
#
#     # exit()
#     ## PLOT regional mass balance
#
#     oce_df = oce_df/1000
#     Aw_oce_obs_df = Aw_oce_obs_df/1000
#     reg_sig = reg_sig/1000
#
#     stats_df = pd.DataFrame(Aw_oce_obs_df, columns=['timeseries']).dropna()
#     # stats_df['sigma'] = stats_df['timeseries'].std()
#     # stats_df['mean'] = stats_df['timeseries'].mean()
#     # std = stats_df['timeseries'].std()
#     # mean = stats_df['timeseries'].mean()
#     # print(stats_df)
#
#     climat_mean= stats_df['timeseries'].loc[(stats_df.index >= 1976)&(stats_df.index <= 2002)].mean()
#     climat_std= stats_df['timeseries'].loc[(stats_df.index >= 1976)&(stats_df.index <= 2002)].std()
#     # print(climat_std)
#
#     stats_df['30yr_mean'] = np.where((stats_df.index >= 1976) & (stats_df.index <= 2002), climat_mean, np.nan)
#     stats_df['30yr_std'] = np.where((stats_df.index >= 1976) & (stats_df.index <= 2002), climat_std, np.nan)
#
#     stats_df['10_year_moving_mean'] = stats_df['timeseries'].rolling(10, min_periods=5).mean()
#
#     stats_df['raw_anomaly'] = stats_df['timeseries'] - climat_mean
#     stats_df['detrended_anomaly'] = stats_df['timeseries'] - stats_df['10_year_moving_mean']
#     std_detrended = stats_df['detrended_anomaly'].std()
#     # print(std_detrended)
#     # exit()
#     stats_df['decadal_extreme_years'] = np.where((stats_df['detrended_anomaly'] < - 2 * std_detrended ) | (stats_df['detrended_anomaly'] > 2 * std_detrended ), stats_df['timeseries'], np.nan )
#     stats_df['clim_extreme_years'] = np.where((stats_df['raw_anomaly'] < - 2 * climat_std ) | (stats_df['raw_anomaly'] > 2 * climat_std ), stats_df['timeseries'], np.nan )
#     # stats_df['clim_extreme_years'] = np.where((stats_df.index >= stats_df.index.min() + 5 ), stats_df['clim_extreme_years'], np.nan )
#
#     # stats_df.to_csv(os.path.join(out_dir, 'stats', region+'_stats.csv'))
#     # exit()
#
#     if end_plot == '_B_reg_and_err_':
#         fig = plt.figure(dpi=300)
#         ax = fig.add_subplot(111)
#
#         # plot regional B
#         ax.plot(Aw_oce_obs_df, color=color[region], linewidth= 2, alpha=0.8)
#         # ax.plot(Aw_oce_obs_df, color=color[region], linewidth= 1.5, alpha=1)
#         # plot regional B uncertainty
#         plt.fill_between(Aw_oce_obs_df.index, Aw_oce_obs_df + reg_sig,
#                          Aw_oce_obs_df - reg_sig, color=color[region], alpha=0.5, linewidth=0)
#         # plt.fill_between(Aw_oce_obs_df.index, Aw_oce_obs_df + reg_sig,
#         #                  Aw_oce_obs_df - reg_sig, color='silver', alpha=0.6, linewidth=0)
#
#         # # plot zemp regional B
#         # if region in ['SA1','SA2']:
#         #     in_zemp_df = in_data_zemp + 'Zemp_etal_results_region_' + rgi_code[region] + '_SAN.csv'
#         # else:
#         #     in_zemp_df = in_data_zemp + 'Zemp_etal_results_region_'+rgi_code[region]+'_'+region+'.csv'
#         #
#         # zemp_df = pd.read_csv(in_zemp_df, encoding='utf-8', delimiter=',', header=26, index_col= 'Year')
#         #
#         # zemp_ba = zemp_df[' INT_mwe']
#         # zemp_sig_ba = zemp_df[' sig_Total_mwe']
#         #
#         # ax.plot(zemp_ba, color='black', linewidth= 2)
#         # plt.fill_between(zemp_ba.index, zemp_ba + zemp_sig_ba,
#         #                  zemp_ba - zemp_sig_ba, color='silver', alpha=0.4, linewidth=0)
#
#         # per_gla_obs= "{:.1f}".format(nb_gla_obs/nb_gla_reg * 100)
#         # ax.set_title('region '+ region, fontsize=12)
#         ax.axhline(0, color='Grey', linewidth=1)
#         # ax.set_xlabel('Year', size=18, weight=600)
#         # ax.set_ylabel(r'$B_{CE}$ (m w.e.)', size=18, weight=600)
#         # ax.text(1990, 3.8, 'N$_{gla}$ = ' + str(nb_gla_reg) + ' glaciers')
#         ax.text(1918, -3.3, rgi_code[region] + ' - ' + rgi_region[region], size=22, weight=600)
#
#         # ## plot extremes
#         # for index, row in stats_df.iterrows():
#         #     color2 = '#107ab0' if row['decadal_extreme_years'] > 0 else '#990f4b'
#         #     x1 = index - 0.5
#         #     x2 = index + 0.5
#         #     y1 = row['decadal_extreme_years']
#         #     y2 = row['clim_extreme_years']
#         #     ax.fill([x1, x1, x2, x2], [0, y1, y1, 0], color=color2, linewidth=1, zorder=10)
#         #     # ax.fill([x1, x1, x2, x2], [0, y2, y2, 0], color='black', linewidth=1, zorder=10)
#         #     ax.fill([x1, x1, x2, x2], [0, y2, y2, 0], fill=None,  hatch= "//////", edgecolor= 'black', alpha = 0.8, linewidth=0.2, zorder=10)
#
#         plt.xlim([1915, plt_year_max])
#         # plt.legend()
#         ax.tick_params(labelsize=18)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#
#         if axes == 'eq_axes':
#             plt.ylim((-3.5, 2.3))
#
#         # plt.legend(loc=3)
#         # plt.tight_layout()
#         # out_plot_dir = out_dir + 'plot'+end_plot+'with_extremes\\'
#         out_plot_dir = os.path.join(out_dir, 'plot'+end_plot)
#         if not os.path.exists(out_plot_dir):
#             os.mkdir(out_plot_dir)
#
#         # set output format
#         # out_fig= out_plot_dir + region +'_fig_Ba_series_Aw_extreme_years.svg'
#         # out_png= out_plot_dir + region +'_fig_Ba_series_Aw_extreme_years.png'
#         out_fig= os.path.join(out_plot_dir, region +'_fig_Ba_series_Aw.svg')
#         out_png= os.path.join(out_plot_dir, region +'_fig_Ba_series_Aw.png')
#
#         plt.savefig(out_fig)
#         plt.savefig(out_png)
#         print('Plot saved as {}.'.format(out_png))
#         # plt.show()
#         plt.close()
#         # exit()
#
#
#     # elif end_plot == '_stats_norm_std_reg_':
#     #     fig = plt.figure( dpi=300)
#     #
#     #     stats_df['norm_detrended_anomaly'] = stats_df['detrended_anomaly'] / std_detrended
#     #     stats_df['norm_raw_anomaly'] = stats_df['raw_anomaly'] / climat_std
#     #
#     #     ax = sns.histplot(stats_df['norm_raw_anomaly'], color='silver', alpha=0.6, bins= 20, kde=True, linewidth=0, label='Raw anomally (1976-2002)')
#     #     sns.histplot(stats_df['norm_detrended_anomaly'], color='indigo', alpha=0.4, bins=20, kde=True, linewidth=0, label='Decadal anomally')
#     #
#     #     sns.rugplot(stats_df['norm_detrended_anomaly'], color='indigo')
#     #     sns.rugplot(stats_df['norm_raw_anomaly'], color='grey')
#     #     ymin, ymax = plt.ylim()
#     #
#     #     plt.axvline(0 , color='black', linestyle='dashed', linewidth=2)
#     #
#     #     # ax.yaxis.set_major_formatter(PercentFormatter(1))
#     #     plt.xlim([-3.5, 3.5])
#     #     # ax.text(-3.4, ymax, rgi_code[region] + '-' + rgi_region[region], size=22, weight=600)
#     #
#     #     ax.tick_params(labelsize=18)
#     #     ax.spines['right'].set_visible(False)
#     #     ax.spines['top'].set_visible(False)
#     #     ax.spines['left'].set_visible(False)
#     #
#     #     ax.set_xlabel('normalized \u03C3', size=16, weight=600)
#     #     ax.set_ylabel('number of years', size=16, weight=600)
#     #     plt.legend(loc='best', fontsize=10, frameon=False)
#     #     plt.tight_layout()
#     #
#     #     out_plot_dir = os.path.join(out_dir, 'per_region_stats', 'plot_' + region + '_B_extremes_stats')
#     #     if not os.path.exists(out_plot_dir):
#     #         os.mkdir(out_plot_dir)
#     #
#     #     # set output format
#     #     out_svg= os.path.join(out_plot_dir, region +'_fig'+end_plot+'.svg')
#     #     out_fig = os.path.join(out_plot_dir, region + '_fig' + end_plot + '.png')
#     #
#     #     plt.savefig(out_svg, dpi=300)
#     #     plt.savefig(out_fig, dpi=300)
#     #     print('Plot saved as {}.'.format(out_fig))
#     #     # plt.show()
#     #     plt.close()
#     #     exit()
#     #
#     # if end_plot == '_stats_B_anomalies_reg_':
#     #     fig, ax = plt.subplots(1, 2, sharey=True, sharex=True , figsize=(8,5), dpi=300)
#     #
#     #     x2 = std_detrended
#     #     x3 = -std_detrended
#     #     x4 = 2 * std_detrended
#     #     x5 = - 2 * std_detrended
#     #     x6 = 3 * std_detrended
#     #     x7 = - 3 * std_detrended
#     #
#     #     z2 = climat_std
#     #     z3 = - climat_std
#     #     z4 = 2*climat_std
#     #     z5 = - 2*climat_std
#     #     z6 = 3*climat_std
#     #     z7 = - 3*climat_std
#     #
#     #     sns.histplot(stats_df['detrended_anomaly'], color='indigo', alpha=0.4, bins=20, kde=True, linewidth=0, ax=ax[0])
#     #     # ax = sns.histplot(stats_df['timeseries'],rug=True, color='#4e2b61', bins= 20, kde=True )
#     #     sns.rugplot(stats_df['detrended_anomaly'], color='indigo', ax=ax[0])
#     #
#     #     sns.histplot(stats_df['raw_anomaly'], color='silver', alpha=0.6, bins= 20, kde=True, linewidth=0, ax=ax[1])
#     #     # ax = sns.histplot(stats_df['timeseries'],rug=True, color='#4e2b61', bins= 20, kde=True )
#     #     sns.rugplot(stats_df['raw_anomaly'], color='grey', ax=ax[1])
#     #
#     #     plt.ylim([0, 14])
#     #     ymin, ymax = plt.ylim()
#     #
#     #     ax[0].axvline(x4 , color='#402A7E', linestyle=':', alpha=0.4, linewidth=1.5)
#     #     ax[0].axvline(x5 , color='#402A7E', linestyle=':', alpha=0.4, linewidth=1.5)
#     #     ax[1].axvline(z4 , color= 'black', linestyle=':', alpha=0.4, linewidth=1.5)
#     #     ax[1].axvline(z5 , color= 'black', linestyle=':', alpha=0.4, linewidth=1.5)
#     #
#     #     ax[0].axvline(0 , color='indigo', linestyle='dashed', linewidth=2)
#     #     ax[1].axvline(0 , color='black', linestyle='dashed', linewidth=2)
#     #
#     #     # ax.yaxis.set_major_formatter(PercentFormatter(1))
#     #     # plt.xlim([-3.5, 3.5])
#     #     # ax.text(-3.4, ymax, rgi_code[region] + '-' + rgi_region[region], size=22, weight=600)
#     #
#     #     ax[0].text(x4 + 0.05, ymax , '2\u03C3', color='#402A7E', size=16, weight=600, alpha=0.4)
#     #     ax[1].text(z4 + 0.05, ymax , '2\u03C3', color='grey', size=16, weight=600, alpha=0.4)
#     #
#     #     ax[0].tick_params(labelsize=16)
#     #     ax[0].spines['right'].set_visible(False)
#     #     ax[0].spines['top'].set_visible(False)
#     #     ax[0].spines['left'].set_visible(False)
#     #     ax[1].tick_params(labelsize=16)
#     #     ax[1].spines['right'].set_visible(False)
#     #     ax[1].spines['top'].set_visible(False)
#     #     ax[1].spines['left'].set_visible(False)
#     #     ax[1].get_yaxis().set_visible(False)
#     #
#     #     ax[0].set_xlabel(r'Decadal B anomally (m w.e. year$^{-1}$)', size=14, weight=600)
#     #     ax[1].set_xlabel(r'B anomally (m w.e. year$^{-1}$)', size=14, weight=600)
#     #     ax[0].set_ylabel('number of years', size=14, weight=600)
#     #
#     #
#     #     plt.tight_layout()
#     #
#     #     out_plot_dir = os.path.join(out_dir, 'extreme_years_Climat_Decadal')
#     #     if not os.path.exists(out_plot_dir):
#     #         os.mkdir(out_plot_dir)
#     #
#     #     # set output format
#     #     out_svg= os.path.join(out_plot_dir, region +'_fig'+end_plot+'.svg')
#     #     out_fig = os.path.join(out_plot_dir, region + '_fig' + end_plot + '.png')
#     #
#     #     plt.savefig(out_svg, dpi=300)
#     #     plt.savefig(out_fig, dpi=300)
#     #     print('Plot saved as {}.'.format(out_fig))
#     #     # plt.show()
#     #     plt.close()
#     #     # exit()
#     #
#     # elif end_plot == '_B_reg_extremes':
#     #     fig = plt.figure(figsize=(8,5), dpi=300)
#     #     ax = fig.add_subplot(111)
#     #
#     #     ax.plot(Aw_oce_obs_df, color='black', linewidth=1, alpha=0.5, label='Annual timeseries')
#     #     ax.plot(stats_df['10_year_moving_mean'], color='indigo', linewidth=3, linestyle='-', alpha=0.6, label='Decadal running mean')
#     #     ax.plot(stats_df['30yr_mean'], color='black', linewidth=1, alpha=0.5, linestyle='--', label='1976-2002 mean')
#     #
#     #     plt.fill_between(stats_df.index, stats_df['10_year_moving_mean'] + std_detrended,
#     #                      stats_df['10_year_moving_mean'] - std_detrended, color='#402A7E', alpha=0.05, linewidth=1)
#     #     plt.fill_between(stats_df.index, stats_df['10_year_moving_mean'] + 2 * std_detrended,
#     #                      stats_df['10_year_moving_mean'] - 2 * std_detrended, color='#464898', alpha=0.05, linewidth=1)
#     #     plt.fill_between(stats_df.index, stats_df['10_year_moving_mean'] + 3 * std_detrended,
#     #                      stats_df['10_year_moving_mean'] - 3 * std_detrended, color='#687EB1', alpha=0.05, linewidth=1)
#     #     plt.fill_between(stats_df.index, stats_df['10_year_moving_mean'] + 4 * std_detrended,
#     #                      stats_df['10_year_moving_mean'] - 4 * std_detrended, color='#91B2CB', alpha=0.05, linewidth=1)
#     #     # plt.fill_between(stats_df.index, stats_df['10_year_moving_mean'] + 5 * std_detrended,
#     #     #                  stats_df['10_year_moving_mean'] - 5 * std_detrended, color='#C0DFE4', alpha=0.1, linewidth=1)
#     #
#     #     if region == 'SA2':
#     #         x = stats_df['10_year_moving_mean'].first_valid_index()
#     #         y2 = stats_df.loc[x,'10_year_moving_mean'] - std_detrended
#     #         y3 = stats_df.loc[x,'10_year_moving_mean'] - 2 * std_detrended
#     #         ax.text(x + 0.5 , y2 + 0.1, '\u03C3', color='#402A7E', size=10, weight=600, alpha=0.5)
#     #         ax.text(x + 0.5 , y3 + 0.1, '2\u03C3', color='#402A7E', size=10, weight=600, alpha=0.4)
#     #
#     #     else:
#     #         x = stats_df['10_year_moving_mean'].first_valid_index()
#     #         y2 = stats_df.loc[x,'10_year_moving_mean'] - std_detrended
#     #         y3 = stats_df.loc[x,'10_year_moving_mean'] - 2 * std_detrended
#     #         y4 = stats_df.loc[x,'10_year_moving_mean'] - 3 * std_detrended
#     #         y5 = stats_df.loc[x,'10_year_moving_mean'] - 4 * std_detrended
#     #         # y6 = stats_df.loc[x,'10_year_moving_mean'] -5 * std_detrended
#     #         ax.text(x + 1 , y2 , '\u03C3', color='#402A7E', size=10, weight=600, alpha=0.5)
#     #         ax.text(x + 1 , y3 , '2\u03C3', color='#402A7E', size=10, weight=600, alpha=0.4)
#     #         ax.text(x + 1 , y4 , '3\u03C3', color='#402A7E', size=10, weight=600, alpha=0.3)
#     #         ax.text(x + 1 , y5 , '4\u03C3', color='#402A7E', size=10, weight=600, alpha=0.2)
#     #         # ax.text(x + 1 , y6 , '5\u03C3', color='#402A7E', size=10, weight=600, alpha=0.1)
#     #
#     #     for index, row in stats_df.iterrows():
#     #         color2 = '#107ab0' if row['decadal_extreme_years'] > 0 else '#990f4b'
#     #         x1 = index - 0.5
#     #         x2 = index + 0.5
#     #         y1 = row['decadal_extreme_years']
#     #         y2 = row['clim_extreme_years']
#     #         ax.fill([x1, x1, x2, x2], [0, y1, y1, 0], color2, alpha=0.8, linewidth=1)
#     #         ax.fill([x1, x1, x2, x2], [0, y2, y2, 0], fill=None, hatch="//////", edgecolor='black', alpha=0.8,
#     #                 linewidth=0.2, zorder=10)
#     #
#     #         if region == 'SA2':
#     #             if np.isfinite(y1):
#     #                 ax.text(x2-2, y1 +0.01, str(index), color=color2, size=10, weight=600, alpha=0.7)
#     #         else:
#     #             if np.isfinite(y1):
#     #                 ax.text(x2, y1, str(index), color=color2, size=10, weight=600, alpha=0.7)
#     #
#     #     ax.set_xlabel('Year', size=16, weight=600)
#     #     ax.set_ylabel(r'Mass balance (m w.e.)', size=16, weight=600)
#     #     ax.axhline(0, color='Grey', linewidth=1)
#     #     xmin, xmax = plt.xlim()
#     #     ax.text(xmin +2, -3.3, rgi_code[region] + '-' + rgi_region[region], size=22, weight=600)
#     #     ax.text(xmin +2, 2, '1976-2002 std: '+ str(round(climat_std,2))+'\nDecadal std: ' +str(round(std_detrended,2)), color= 'black',size=10)
#     #     # plt.xlim([1915, plt_year_max])
#     #     ax.tick_params(labelsize=18)
#     #     ax.spines['right'].set_visible(False)
#     #     ax.spines['top'].set_visible(False)
#     #     ax.spines['left'].set_visible(False)
#     #     plt.ylim((-3.5, 2.3))
#     #     plt.legend(loc='best', fontsize=10, frameon=False)
#     #     plt.tight_layout()
#     #
#     #     out_plot_dir = os.path.join(out_dir, 'extreme_years_Climat_Decadal')
#     #     if not os.path.exists(out_plot_dir):
#     #         os.mkdir(out_plot_dir)
#     #
#     #     # set output format
#     #     out_svg = os.path.join(out_plot_dir, region + '_fig' + end_plot + '_years_Climat_Decadal.svg')
#     #     out_fig = os.path.join(out_plot_dir, region + '_fig' + end_plot + '_years_Climat_Decadal.png')
#     #     plt.savefig(out_svg)
#     #     plt.savefig(out_fig)
#     #     print('Plot saved as {}.'.format(out_fig))
#     #     # plt.show()
#     #     plt.close(0)
#     #     # exit()
#
#     # if end_plot == '_OCE_and_B_reg_':
#     #     fig = plt.figure(dpi=300)
#     #     ax = fig.add_subplot(111)
#     #     # # plot regional individual OCEs
#     #     for item in oce_df.columns:
#     #         ax.plot(oce_df.index, oce_df[item], color=color[region], alpha=0.5, linewidth=0.5)
#     #
#     #     # plot regional B
#     #     ax.plot(Aw_oce_obs_df, color=color[region], linewidth= 2, alpha=0.8)
#     #     # # plot zemp regional B
#     #     # if region in ['SA1','SA2']:
#     #     #     in_zemp_df = in_data_zemp + 'Zemp_etal_results_region_' + rgi_code[region] + '_SAN.csv'
#     #     # else:
#     #     #     in_zemp_df = in_data_zemp + 'Zemp_etal_results_region_'+rgi_code[region]+'_'+region+'.csv'
#     #     #
#     #     # zemp_df = pd.read_csv(in_zemp_df, encoding='utf-8', delimiter=',', header=26, index_col= 'Year')
#     #     #
#     #     # zemp_ba = zemp_df[' INT_mwe']
#     #     # zemp_sig_ba = zemp_df[' sig_Total_mwe']
#     #     #
#     #     # ax.plot(zemp_ba, color='black', linewidth= 2)
#     #     # plt.fill_between(zemp_ba.index, zemp_ba + zemp_sig_ba,
#     #     #                  zemp_ba - zemp_sig_ba, color='silver', alpha=0.4, linewidth=0)
#     #
#     #     per_gla_obs= "{:.1f}".format(nb_gla_obs/nb_gla_reg * 100)
#     #     # ax.set_title('region '+ region, fontsize=12)
#     #     ax.axhline(0, color='Grey', linewidth=1)
#     #     # ax.set_xlabel('Year', size=18, weight=600)
#     #     # ax.set_ylabel(r'$B_{CE}$ (m w.e.)', size=18, weight=600)
#     #     # ax.text(1990, 3.8, 'N$_{gla}$ = ' + str(nb_gla_reg) + ' glaciers')
#     #     ax.text(1918, -3.3, rgi_code[region] + '-' + rgi_region[region], size=22, weight=600)
#     #
#     #     plt.xlim([1915, plt_year_max])
#     #     # plt.legend()
#     #     ax.tick_params(labelsize=18)
#     #     ax.spines['right'].set_visible(False)
#     #     ax.spines['top'].set_visible(False)
#     #     ax.spines['left'].set_visible(False)
#     #
#     #     if axes == 'eq_axes':
#     #         plt.ylim((-3.5, 2.3))
#     #
#     #     # plt.legend(loc=3)
#     #     # plt.tight_layout()
#     #     out_plot_dir = out_dir + 'plot'+end_plot+'\\'
#     #     if not os.path.exists(out_plot_dir):
#     #         os.mkdir(out_plot_dir)
#     #
#     #     # set output format
#     #     out_svg= out_plot_dir + region +'_fig'+end_plot+'.svg'
#     #     out_fig= out_plot_dir + region +'_fig'+end_plot+'.png'
#     #
#     #     plt.savefig(out_svg)
#     #     plt.savefig(out_fig)
#     #     print('Plot saved as {}.'.format(out_png))
#     #     # plt.show()
#     #     plt.close()
#     #     # exit()
#     #
#
#
# Reg_mb_df = Reg_mb_df.loc[(Reg_mb_df.index >= DM_series_min_yr)] / 1000
# Reg_sig_mb_df = Reg_sig_mb_df.loc[(Reg_sig_mb_df.index >= DM_series_min_yr)] / 1000
# ### Save regional Mass balance series
# Reg_mb_df.to_csv(os.path.join(out_dir, 'Regional_B_series_AreaWeighted.csv'))
# Reg_sig_mb_df.to_csv(os.path.join(out_dir, 'Regional_B_series_uncertainty.csv'))
#

print('.........................................................................................')
print('"The End"')
print('.........................................................................................')
exit()
