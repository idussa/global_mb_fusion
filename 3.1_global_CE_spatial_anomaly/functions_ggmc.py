"""functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

def create_mb_dataframe(in_df, id_lst, yr_lst, mb_field):
    """This function selects mass-balance data from the input file into a dataframe."""
    print('Creating dataframe of {}...'.format(mb_field))

    mb_dict = {}

    for id in id_lst:
        mb_lst = []
        for yr in yr_lst:
            if yr in in_df.loc[in_df['WGMS_ID'] == id]['YEAR'].values:
                mb_lst.append(
                    in_df[(in_df['WGMS_ID'] == id) & (in_df['YEAR'] == yr)].iloc[0][mb_field])
            else:
                mb_lst.append(np.nan)
        mb_dict.update({id: mb_lst})

    mb_df = pd.DataFrame(mb_dict, index=yr_lst)

    print('..done.')
    return mb_df

def calc_anomalies(in_df, ref_period, region):
    """This function calculates the anomalies of glacier mass balances over a defined reference period."""
    print('Calculating anomalies for reference period from {} to {}...'.format(min(ref_period), max(ref_period)))

    # create subset over reference period
    in_ref_df = in_df.loc[in_df.index.isin(ref_period)]

    # create subset with minimal data
    ref_period_df = in_df.loc[in_df.index.isin(ref_period)]
    if region == 'ASN':
        ref_period_ids = ref_period_df.count() > 3
    else:
        ref_period_ids = ref_period_df.count() > 7
    ref_period_id_lst = list(ref_period_ids[ref_period_ids].index)

    # create subset of glacier ids with good data over reference period
    good_ids_in_ref_df = in_ref_df[ref_period_id_lst]
    good_ids_in_df=in_df[ref_period_id_lst]

    # calculate anomaly (x_i-x_avg) for data over reference period
    avg_ref_df = good_ids_in_ref_df.mean()
    anomaly_ref_df = round(good_ids_in_df - avg_ref_df, 0)
    print(anomaly_ref_df)
    print('done.')
    return anomaly_ref_df

def calc_anomalies_unc(in_df, in_unc_df, ref_period, region):
    """This function calculates the uncertainties of glacier mass balances series"""

    # create subset with minimal data
    ref_period_df = in_df.loc[in_df.index.isin(ref_period)]
    if region == 'ASN':
        ref_period_ids = ref_period_df.count() > 3
    else:
        ref_period_ids = ref_period_df.count() > 7
    ref_period_id_lst = list(ref_period_ids[ref_period_ids].index)

    # calculate mb uncertainty of glacier ids with good data over reference period
    unc_ref_df = in_unc_df[ref_period_id_lst]
    reg_unc_mean= np.nanmean(unc_ref_df)

    for id in ref_period_id_lst:
        year_min = in_df[id].first_valid_index()
        yrs= list(range(1885,year_min))
        if unc_ref_df[id].isnull().all():
            unc_ref_df[id][id].fillna(reg_unc_mean, inplace=True)
        else:
            unc_ref_df[id].fillna(unc_ref_df[id].mean(), inplace=True)
        # unc_ref_df.loc[unc_ref_df.index.isin(yrs), [id]] = np.nan
        unc_ref_df[id].mask(unc_ref_df.index.isin(yrs), np.nan, inplace=True)

    print('done.')
    return unc_ref_df


def plot_reg_anomalies(in_ba_df_lst, path):
    """This function plots the regional mass-balance series and volcanic eruptions over time."""
    print('Plotting mass-balance time series...')

    # set output format
    file_type = '.svg'

    # define name of figure
    out_fig = path + '\\out_data\\Fig_regional_mass-balance_timeseries_' + file_type

    # define glacier region labels
    reg_lst = ['(a) ALA', '(b) WNA', '(c) ACN', '(d) ACS', '(e) GRL',
               '(f) ISL', '(g) SJM', '(h) SCA', '(i) RUA', '(j) ASN',
               '(k) CEU', '(l) CAU', '(m) ASC', '(n) ASW', '(o) ASE',
               '(p) TRP', '(q) SAN', '(r) NZL', '(s) ANT', '(t) Global mean of regional averages']

    # initialize plot
    # fig, ax = plt.subplots(20, 3, sharex='col', figsize=(8, 24))
    fig, ax = plt.subplots(20, 1, sharex='col', figsize=(24, 24))

    # plot regional and global mass-balance series
    for i in range(0, len(in_ba_df_lst)):
        ax[i].axhline(0, color='black', linewidth=0.5, zorder=1)

        # plot annual values
        ax[i].bar(in_ba_df_lst[i].index, in_ba_df_lst[i] / 1000, color='black', zorder=4)

        # set labels
        ax[i].set_ylim(-5, 4.5)
        ax[i].set_xlim(1960, 2020)
        ax[i].set_yticks(range(-3, 4, 3))
        ax[i].text(0.005, 0.85, str('{}'.format(reg_lst[i])), fontsize='small', ha='left', va='center',
                   transform=ax[i].transAxes)
        ax[i].set_ylabel('m w.e.', fontsize='small')


    # save plot
    # plt.tight_layout()
    plt.savefig(out_fig)

    print('Plot saved as {}.'.format(out_fig))

    # clear plot
    plt.clf()

    return

def plot_gla_oce(in_cal_series_df, in_geo_df, glacier_id, distance, max_year, min_lenght_geo, region, run, gla_dir):
    #add in_geo_df to arguments
    """This function plots the anomalies of glacier mass balances over a defined reference period."""
    print('Plotting calibrated series for glacier = {}'.format(glacier_id))

    # set output format
    file_type = '.svg'

    # define name of figure
    out_fig = gla_dir + '\\'+str(region) +'_glacier_id_'+str(glacier_id)+ file_type

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot zero balance line
    ax.axhline(0, color='Grey', linewidth=1)


    in_cal_series_df = in_cal_series_df/1000


    # plot geodetic mass change trends
    for index, row in in_geo_df.iterrows():
        color = 'b' if row['mb_chg_rate'] > 0 else 'r'
        x1 = row['ini_date']
        x2 = row['fin_date']
        y1 = row['mb_chg_rate']/1000
        y2 = row['sigma_tot_mb_chg']
        ax.fill([x1, x1, x2, x2], [0, y1, y1, 0], color, alpha=0.1)
        ax.plot([x1, x2], [y1, y1], color, linewidth=0.75, alpha=0.8, solid_capstyle='butt')

        # ax.fill([x1, x1, x2, x2], [y1+y2, y1-y2, y1-y2, y1+y2], color='grey', alpha=0.1)
        # ax.plot([x1, x2], [y1, y1], color, linewidth=0.75, alpha=0.8, solid_capstyle='butt')

    # plot calibrated glaciological series
    # print(in_cal_series_df)
    for item in in_cal_series_df.columns:
        ax.plot(in_cal_series_df.index, in_cal_series_df[item], color='Silver', linewidth=0.5)

    # plot average calibrated series
    # ax.plot(in_cal_series_df['normal_MEAN'], color='coral', label= 'Arithmetic Mean', linewidth=1)
    # ax.plot(in_cal_series_df['weighted_MEAN'], color='Black', label= 'Weighted Mean (Wu +Wd)', linewidth=1.2)

    # set labels
    # ax.set_title('Calibrated '+ run +' mass-change \n' + glacier_name + ', WGMS Id = {}'.format(wgms_id),
    #              fontsize='medium')
    # ax.set_title('Observational calibrated estimate \n Glacier WGMS Id '+str(glacier_id)+'  - Regional anomaly '+str(region),
    #              fontsize='medium')
    # ax.set_title('Observational Consensus Estimate \n' + glacier_name + ' glacier - glacier anomaly',
    #              fontsize='medium')

    ax.set_xlabel('Year', size=18, weight=600)
    ax.set_ylabel(r'$B_{calibrated}$ (m w.e.)', size=18, weight=600)
    ax.set_xticks(np.arange(min(in_cal_series_df.index), max(in_cal_series_df.index) + 2, step=10))


    # save plot
    ax.tick_params(labelsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim([1950, max_year])
    plt.xticks(np.arange(1960, max_year, 20))


    ax.set_ylim([-3.5, 2])
    # plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print('Plot saved as {}.'.format(out_fig))
    # plt.show()

    # clear plot
    plt.close()

    return

def plot_gla_oce_and_unc(in_cal_series_df, in_oce_df, in_geo_df, in_oce_unc_df,  glacier_id, min_year, max_year, min_lenght_geo, region, run, gla_dir):
    #add in_geo_df to arguments
    """This function plots the anomalies of glacier mass balances over a defined reference period."""
    print('Plotting calibrated series for glacier = {}'.format(glacier_id))

    # set output format
    file_type = '.svg'

    # define name of figure
    out_fig = gla_dir + '\\Fig2_OCE_glacier_id_'+str(glacier_id)+'_region_'+str(region) + file_type

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot zero balance line
    ax.axhline(0, color='Grey', linewidth=1)

    in_cal_series_df = in_cal_series_df/1000
    in_oce_unc_df = in_oce_unc_df/1000
    in_oce_df = in_oce_df/1000

    # plot geodetic mass change trends
    for index, row in in_geo_df.iterrows():
        color = 'b' if row['mb_chg_rate'] > 0 else 'r'
        x1 = row['ini_date']
        x2 = row['fin_date']
        y1 = row['mb_chg_rate']/1000
        ax.plot([x1, x2], [y1, y1], color, linewidth=0.75, alpha=0.8, solid_capstyle='butt')


    # plot calibrated glaciological series
    for item in in_cal_series_df.columns:
        ax.plot(in_cal_series_df.index, in_cal_series_df[item], color='Silver', linewidth=0.5)

    plt.fill_between(in_cal_series_df.index, in_oce_df + in_oce_unc_df[glacier_id],
                     in_oce_df - in_oce_unc_df[glacier_id], color='grey', alpha=0.3, linewidth=0)

    # plot average calibrated series
    ax.plot(in_oce_df, color='black', linewidth=1)

    # set labels
    # ax.set_title('Calibrated '+ run +' mass-change \n' + glacier_name + ', WGMS Id = {}'.format(wgms_id),
    #              fontsize='medium')
    # ax.set_title('Observational calibrated estimate \n Glacier WGMS Id '+str(glacier_id)+'  - Regional anomaly '+str(region),
    #              fontsize='medium')
    # ax.set_title('Observational Consensus Estimate \n' + glacier_name + ' glacier - glacier anomaly',
    #              fontsize='medium')

    ax.set_xlabel('Year', size=18, weight=600)
    ax.set_ylabel(r'$B_{CE}$ (m w.e.)', size=18, weight=600)
    plt.xticks(np.arange(1960, max_year, 20))

    # save plot
    ax.tick_params(labelsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim([1950, max_year])
    plt.xticks(np.arange(1960, max_year, 20))

    ax.set_ylim([-3.5, 2])
    # plt.legend(loc=3)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print('Plot saved as {}.'.format(out_fig))
    # plt.show()

    # clear plot
    plt.close()

    return

def date_format(month,year):
    default_month = 6.0
    month=default_month if month > 12.0 else month
    offset = year + (1.0 / 12 * (month - 1))
    return offset

def cum_to_rate(elev_chg_cum,sur_date,ref_date):
    if (sur_date-ref_date) != 0:
        elev_chg_rate = elev_chg_cum/(sur_date-ref_date) if elev_chg_cum != 0 else elev_chg_cum
        return elev_chg_rate
    else:
        return elev_chg_cum

def dis_fil(row, ini_date, fin_date):
    if row > fin_date:
        return row - fin_date + 1
    if row <= ini_date:
        return ini_date + 2 - row
    else:
        return 1.0

def convert_date_time_to_decimal_date(date_time):
    """
This function converts a date and a time to a decimal date value
Inputs:
- date_time: datetime object

Outputs:
- decimal_date_float: float

"""
    hourdec = (date_time.hour + date_time.minute / 60. + date_time.second / 3600.) / 24.
    doy = date_time.timetuple().tm_yday
    decimal_date = date_time.year + (doy + hourdec) / 365.25
    decimal_date = float('{:.8f}'.format(decimal_date))
    return decimal_date


def convert_decimal_date_to_date_time(decimal_date):
    """
This function converts a decimal date and a date and time
Inputs:
- decimal_date: float

Outputs:
- date_time: datetime object
- date_time_string: formated string from the datetime object
"""
    decimal_date = float('{:.8f}'.format(decimal_date))
    year = np.floor(decimal_date)
    decimal_day = (decimal_date - np.floor(decimal_date)) * 365.25
    doy = np.floor(decimal_day)
    decimal_time = (decimal_day - doy) * 24.
    hour = np.floor(decimal_time)
    minute = np.floor((decimal_time - hour) * 60.)
    second = (decimal_time - hour - minute / 60.) * 3660.
    raw_str = str(int(year)) + '{0:03d}'.format(int(doy)) + '{0:02d}'.format(int(hour)) + '{0:02d}'.format(
        int(minute)) + '{0:02d}'.format(int(second))
    date_time = datetime.datetime.strptime(raw_str, '%Y%j%H%M%S')
    date_time_string = date_time.strftime('%Y-%m-%d %H:%M:%S')
    print(date_time_string)
    return date_time, date_time_string
