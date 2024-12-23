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
from propagation_functions_ram import wrapper_latlon_double_sum_covar, sig_dh_spatialcorr, sig_rho_dv_spatialcorr, ba_anom_spatialcorr
# from propagation_functions import wrapper_latlon_double_sum_covar, sig_dh_spatialcorr, sig_rho_dv_spatialcorr, ba_anom_spatialcorr

from scipy.stats import norm
pd.options.mode.chained_assignment = None  # default='warn'
import scipy as sp  # usage: scipy.stats.funct()

reg_lst= ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']

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

path_oce = os.path.join(path_proj, '3.3_Kriging_global_CE_spatial_anomaly', 'out_data_'+fog_version+'_review','OCE_files_by_region') # path to regional OCE files

#################################################################################################
##    Define parameters
#################################################################################################
out_dir = os.path.join(path, 'out_data_'+fog_version+'_review')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

###################################################################################
## If not done before: Bring files together
Reg_mb_lst = []
Reg_sig_mb_lst = []

for region in reg_lst:
    in_data = os.path.join(out_dir, region +'_B_and_sigma.csv')
    data_df = pd.read_csv(in_data, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
    mb_df = pd.DataFrame(data_df['Aw_B m w.e.'])
    mb_df = mb_df.rename(columns={'Aw_B m w.e.': region})
    sig_df = pd.DataFrame(data_df['sigma_B m w.e.'])
    sig_df = sig_df.rename(columns={'sigma_B m w.e.': region})
    Reg_mb_lst.append(mb_df)
    Reg_sig_mb_lst.append(sig_df)

Reg_mb_df = pd.concat(Reg_mb_lst, axis = 1)
Reg_sig_mb_df = pd.concat(Reg_sig_mb_lst, axis = 1)

### Save regional Mass balance series
Reg_mb_df.to_csv(os.path.join(out_dir, 'Regional_B_series_AreaWeighted.csv'))
Reg_sig_mb_df.to_csv(os.path.join(out_dir, 'Regional_B_series_uncertainty.csv'))

###################################################################################

DM_series_min_yr = 1915 # starting year of DM series saved in the .csv files

# end_plot= '_OCE_and_B_reg_' # plots Regional Mass balance and the individual region OCE
end_plot = '_B_reg_and_err_' # plots Regional Mass balance and the relative mass balance uncertainty

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
    # print(Aw_oce_obs_df)

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

    # exit()
    ####### Calculate Regional specific mass balance time series ##########

    Reg_mb_df[region] = Aw_oce_obs_df
    nb_unobs_gla = nb_gla_reg - nb_gla_obs

    # # Fully correlated propagation for residual error of anomaly
    # Aw_sig_anom_obs_df = rel_sig_anom_mb_df.sum(axis=1, min_count=1)

    # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
    # to compute the pairwise error propagation for dh and density across all glaciers of that year
    list_sig_dh_yearly = []
    list_sig_rho_yearly = []
    list_sig_anom_yearly = []

    for i in range(len(rel_sig_dh_mb_df.index)):

        print(f"Propagating uncertainties from glaciers to region for year {rel_sig_dh_mb_df.index[i]}")

        # Create dataframe with dh errors, lat and lon
        yearly_dh_df = rel_sig_dh_mb_df.iloc[i, :]
        yearly_dh_df["errors"] = yearly_dh_df.values.flatten()
        yearly_dh_df["lat"] = np.array(list_lat)
        yearly_dh_df["lon"] = np.array(list_lon)

        # Spatial correlations for dh
        sig_dh_obs = wrapper_latlon_double_sum_covar(yearly_dh_df, spatialcorr_func=sig_dh_spatialcorr)

        # Check propagation works as intended: final estimate is between fully correlated and independent
        sig_dh_fullcorr = np.sum(yearly_dh_df["errors"])
        sig_dh_uncorr = np.sqrt(np.sum(yearly_dh_df["errors"]**2))
        print(f"{sig_dh_uncorr}, {sig_dh_obs}, {sig_dh_fullcorr}")
        assert sig_dh_uncorr <= sig_dh_obs <= sig_dh_fullcorr

        # Create dataframe with rho errors, lat and lon
        yearly_rho_df = rel_sig_rho_mb_df.iloc[i, :]
        yearly_rho_df["errors"] = yearly_rho_df.values
        yearly_rho_df["lat"] = np.array(list_lat)
        yearly_rho_df["lon"] = np.array(list_lon)

        # Spatial correlation for rho for a 1-year period
        def sig_rho_dv_spatialcorr_yearly(d):
            return sig_rho_dv_spatialcorr(d, dt=1)
        sig_rho_obs = wrapper_latlon_double_sum_covar(yearly_rho_df, spatialcorr_func=sig_rho_dv_spatialcorr_yearly)

        # Check propagation works as intended: final estimate is between fully correlated and independent
        sig_rho_fullcorr = np.sum(yearly_rho_df["errors"])
        sig_rho_uncorr = np.sqrt(np.sum(yearly_rho_df["errors"] ** 2))
        # print(f"{sig_rho_uncorr}, {sig_rho_obs}, {sig_rho_fullcorr}")
        assert sig_rho_uncorr <= sig_rho_obs <= sig_rho_fullcorr

        # Create dataframe with anom errors, lat and lon
        yearly_anom_df = rel_sig_anom_mb_df.iloc[i, :]
        yearly_anom_df["errors"] = yearly_anom_df.values
        yearly_anom_df["lat"] = np.array(list_lat)
        yearly_anom_df["lon"] = np.array(list_lon)

        # Spatial correlations for anom
        sig_anom_obs = wrapper_latlon_double_sum_covar(yearly_anom_df, spatialcorr_func=ba_anom_spatialcorr)

        # Check propagation works as intended: final estimate is between fully correlated and independent
        sig_anom_fullcorr = np.sum(yearly_anom_df["errors"])
        sig_anom_uncorr = np.sqrt(np.sum(yearly_anom_df["errors"]**2))
        # print(f"{sig_anom_uncorr}, {sig_anom_obs}, {sig_anom_fullcorr}")
        assert sig_anom_uncorr <= sig_anom_obs <= sig_anom_fullcorr

        # Append to list for each yearly period
        list_sig_dh_yearly.append(sig_dh_obs)
        list_sig_rho_yearly.append(sig_rho_obs)
        list_sig_anom_yearly.append(sig_anom_obs)

    # And write back the 1D list of uncertainties into an indexed (by YEAR) dataframe
    Aw_sig_dh_obs_df =  pd.DataFrame(index=sig_anom_oce_df.index.copy())
    Aw_sig_dh_obs_df['dh']= list_sig_dh_yearly

    Aw_sig_rho_obs_df = pd.DataFrame(index=sig_anom_oce_df.index.copy())
    Aw_sig_rho_obs_df['rho'] = list_sig_rho_yearly

    Aw_sig_anom_obs_df = pd.DataFrame(index=sig_anom_oce_df.index.copy())
    Aw_sig_anom_obs_df['anom']= list_sig_anom_yearly

    # print(Aw_sig_dh_obs_df)
    # print(Aw_sig_rho_obs_df)
    # print(Aw_sig_anom_obs_df)

    Sig_oce_obs_propag = np.sqrt(Aw_sig_dh_obs_df['dh']**2 + Aw_sig_rho_obs_df['rho']**2 + Aw_sig_anom_obs_df['anom']**2)
    # print(Sig_oce_obs_propag)
    # exit()

    # Defining area-weighted uncertainty of unobserved glaciers based on the mean uncertainty of observed glaciers
    area_unobs = round(tot_area_rgi_reg, 2) - round(tot_area_obs, 2)
    sig_W_unobs = Sig_oce_obs_gla * (area_unobs / tot_area_rgi_reg)

    # Area-weight the observed glaciers before combining in final uncertainty
    sig_W_obs = Sig_oce_obs_propag * (tot_area_obs / tot_area_rgi_reg)

    # Final regional uncertainty!
    reg_sig = np.sqrt(sig_W_obs**2 + sig_W_unobs**2)

    Reg_sig_mb_df[region] = reg_sig

    Reg_sumary_df['Aw_B m w.e.'] = Aw_oce_obs_df/1000
    Reg_sumary_df['sigma_B m w.e.'] = reg_sig / 1000
    Reg_sumary_df['sigma_propagated m w.e.'] = Sig_oce_obs_propag / 1000
    Reg_sumary_df['sigma_dh m w.e.'] = Aw_sig_dh_obs_df / 1000
    Reg_sumary_df['sigma_rho m w.e.'] = Aw_sig_rho_obs_df / 1000
    Reg_sumary_df['sigma_anom m w.e.'] = Aw_sig_anom_obs_df / 1000
    Reg_sumary_df.to_csv(os.path.join(out_dir, region +'_B_and_sigma.csv'))
    # exit()

    # reg_sig_asw= pd.read_csv(os.path.join(out_dir, 'ASW_B_and_sigma.csv'), encoding='latin1', delimiter=',', header=0, index_col= 'YEAR')
    # reg_sig_asc= pd.read_csv(os.path.join(out_dir, 'Regional_B_series_AreaWeighted_edited.csv'), encoding='latin1', delimiter=',', header=0, index_col= 'YEAR')
    # Aw_oce_obs_df = reg_sig_asc['ASC']
    # reg_sig = reg_sig_asw['sigma_B m w.e.']

    # exit()
    ## PLOT regional mass balance

    oce_df = oce_df/1000
    Aw_oce_obs_df = Aw_oce_obs_df/1000
    reg_sig = reg_sig/1000

    if end_plot == '_B_reg_and_err_':
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)

        # plot regional B
        ax.plot(Aw_oce_obs_df, color=color[region], linewidth= 2, alpha=0.8)

        # plot regional B uncertainty
        plt.fill_between(Aw_oce_obs_df.index, Aw_oce_obs_df + reg_sig,
                         Aw_oce_obs_df - reg_sig, color=color[region], alpha=0.5, linewidth=0)

        ax.axhline(0, color='Grey', linewidth=1)
        ax.text(1918, -3.3, rgi_code[region] + ' - ' + rgi_region[region], size=22, weight=600)

        plt.xlim([1915, plt_year_max])
        # plt.legend()
        ax.tick_params(labelsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        if axes == 'eq_axes':
            plt.ylim((-3.5, 2.3))

        # plt.legend(loc=3)
        # plt.tight_layout()
        # out_plot_dir = out_dir + 'plot'+end_plot+'with_extremes\\'
        out_plot_dir = os.path.join(out_dir, 'plot'+end_plot)
        if not os.path.exists(out_plot_dir):
            os.mkdir(out_plot_dir)

        # set output format
        out_fig= os.path.join(out_plot_dir, region +'_fig_Ba_series_Aw.svg')
        # out_png= os.path.join(out_plot_dir, region +'_fig_Ba_series_Aw.png')

        plt.savefig(out_fig)
        plt.savefig(out_png)
        print('Plot saved as {}.'.format(out_png))
        # plt.show()
        plt.close()
        # exit()

Reg_mb_df = Reg_mb_df.loc[(Reg_mb_df.index >= DM_series_min_yr)] / 1000
Reg_sig_mb_df = Reg_sig_mb_df.loc[(Reg_sig_mb_df.index >= DM_series_min_yr)] / 1000
### Save regional Mass balance series
Reg_mb_df.to_csv(os.path.join(out_dir, 'Regional_B_series_AreaWeighted.csv'))
Reg_sig_mb_df.to_csv(os.path.join(out_dir, 'Regional_B_series_uncertainty.csv'))


print('.........................................................................................')
print('"The End"')
print('.........................................................................................')
exit()
